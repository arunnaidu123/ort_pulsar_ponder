#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../producers/PulsarUdpReceptor.h"
#include "../producers/RcptRingBuffer.h"
#include "../utils/FilterbankHeader.h"
#include "SsePower.h"

class PulsarStreamWorker
{
public:
    PulsarStreamWorker(unsigned nchans, unsigned nacc)
        : _nchans(nchans)
        , _nacc(nacc)
        , _data_tmp(2 * nchans, 0.0f)
        , _time_buf(1024)
        , _fft_buf(1024)
        , _fft_buf_nchans(nchans + 1)
        , _power_buf(nchans)
        , _plan(nullptr)
        , _plan_nchans(nullptr)
    {
        _plan = fftwf_plan_dft_1d(
            1024,
            reinterpret_cast<fftwf_complex*>(_fft_buf.data()),
            reinterpret_cast<fftwf_complex*>(_time_buf.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE);
        _plan_nchans = fftwf_plan_dft_r2c_1d(
            2 * nchans,
            _data_tmp.data(),
            reinterpret_cast<fftwf_complex*>(_fft_buf_nchans.data()),
            FFTW_MEASURE);

        if(!_plan || !_plan_nchans)
            throw std::runtime_error("Unable to create FFTW plans");
    }

    ~PulsarStreamWorker()
    {
        if(_plan) fftwf_destroy_plan(_plan);
        if(_plan_nchans) fftwf_destroy_plan(_plan_nchans);
    }

    PulsarStreamWorker(const PulsarStreamWorker&) = delete;
    PulsarStreamWorker& operator=(const PulsarStreamWorker&) = delete;

    void process(const FrequencySeries<int8_t>& data_in,
                 std::vector<float>& data_out)
    {
        const std::size_t input_spectra = data_in.size() / 512;
        _data_tmp.resize(input_spectra * 1024);

        for(std::size_t spectrum = 0; spectrum < input_spectra; ++spectrum)
        {
            for(unsigned channel = 0; channel < 512; ++channel)
            {
                const auto& input = *(data_in.cbegin() + spectrum * 512 + channel);
                const std::complex<float> value(input.real(), input.imag());
                _fft_buf[512 + channel] = value;
                _fft_buf[511 - channel] = std::conj(value);
            }

            fftwf_execute(_plan);
            for(unsigned sample = 0; sample < 1024; ++sample)
                _data_tmp[spectrum * 1024 + sample] = _time_buf[sample].real();
        }

        const std::size_t channelized_spectra =
            (input_spectra * 1024) / (2 * _nchans);
        const std::size_t output_spectra = channelized_spectra / _nacc;
        data_out.resize(output_spectra * _nchans);

        for(std::size_t group = 0; group < output_spectra; ++group)
        {
            std::fill(_power_buf.begin(), _power_buf.end(), 0.0f);
            for(unsigned n = 0; n < _nacc; ++n)
            {
                const std::size_t spectrum = group * _nacc + n;
                fftwf_execute_dft_r2c(
                    _plan_nchans,
                    _data_tmp.data() + 2 * _nchans * spectrum,
                    reinterpret_cast<fftwf_complex*>(_fft_buf_nchans.data()));
                accumulate_power_sse4(
                    _fft_buf_nchans.data(), _power_buf.data(), _nchans);
            }

            normalize_power_sse4(
                _power_buf.data(), _nchans, 1.0f / static_cast<float>(_nacc));
            std::copy(_power_buf.begin(), _power_buf.end(),
                      data_out.begin() + group * _nchans);
        }
    }

private:
    unsigned _nchans;
    unsigned _nacc;
    std::vector<float> _data_tmp;
    std::vector<std::complex<float>> _time_buf;
    std::vector<std::complex<float>> _fft_buf;
    std::vector<std::complex<float>> _fft_buf_nchans;
    std::vector<float> _power_buf;
    fftwf_plan _plan;
    fftwf_plan _plan_nchans;
};

class PulsarStreamProcessor
{
public:
    PulsarStreamProcessor(unsigned nchans, unsigned nacc, unsigned number_of_threads)
        : _nchans(nchans)
        , _nacc(nacc)
        , _number_of_threads(number_of_threads)
        , _data_out(number_of_threads)
    {
        _workers.reserve(number_of_threads);
        for(unsigned i = 0; i < number_of_threads; ++i)
            _workers.emplace_back(new PulsarStreamWorker(nchans, nacc));
    }

    void run(RcptRingBuffer<int8_t>& rcpt_buf, std::ofstream& out_file)
    {
        using RingBufferItem = RcptRingBuffer<int8_t>::BufferType;

        ort::ponder::utils::FilterbankHeader<float> fil_header(_nchans);
        bool header_written = false;
        std::vector<std::shared_ptr<RingBufferItem>> buffers(_number_of_threads);
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> worker_errors(_number_of_threads);
        threads.reserve(_number_of_threads);

        while(!rcpt_buf.abort())
        {
            const auto start = std::chrono::high_resolution_clock::now();
            unsigned batch_size = 0;

            try
            {
                for(; batch_size < _number_of_threads; ++batch_size)
                {
                    buffers[batch_size] = rcpt_buf.get_readable();
                    if(!header_written)
                    {
                        const auto& data = *buffers[batch_size]->second;
                        const long double tstart =
                            static_cast<long double>(data.mjd_day()) +
                            (static_cast<long double>(data.mjd_seconds()) +
                             static_cast<long double>(data.mjd_nanoseconds()) * 1e-9L) /
                                86400.0L;
                        fil_header.tstart(tstart);
                        fil_header.tsamp(_nacc * 2.0 * _nchans * 1e-8);
                        fil_header.write_header(out_file);
                        header_written = true;
                    }
                }
            }
            catch(const std::runtime_error&)
            {
                if(batch_size == 0) break;
            }

            threads.clear();
            std::fill(worker_errors.begin(), worker_errors.end(), nullptr);
            for(unsigned i = 0; i < batch_size; ++i)
            {
                threads.emplace_back([this, &buffers, &worker_errors, i] {
                    try
                    {
                        _workers[i]->process(*buffers[i]->second, _data_out[i]);
                    }
                    catch(...)
                    {
                        worker_errors[i] = std::current_exception();
                    }
                });
            }
            for(auto& thread : threads) thread.join();
            for(unsigned i = 0; i < batch_size; ++i)
                if(worker_errors[i]) std::rethrow_exception(worker_errors[i]);

            for(unsigned i = 0; i < batch_size; ++i)
            {
                out_file.write(reinterpret_cast<const char*>(_data_out[i].data()),
                               _data_out[i].size() * sizeof(float));
                buffers[i].reset();
            }
            if(!out_file) throw std::runtime_error("Error writing filterbank output");

            const auto elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start);
            std::cout << "processed " << batch_size << " buffers in "
                      << elapsed.count() << " s\n";
        }
    }

private:
    unsigned _nchans;
    unsigned _nacc;
    unsigned _number_of_threads;
    std::vector<std::vector<float>> _data_out;
    std::vector<std::unique_ptr<PulsarStreamWorker>> _workers;
};

static bool parse_unsigned(const std::string& value, unsigned& result)
{
    std::size_t parsed_characters = 0;
    unsigned long parsed_value = 0;
    try
    {
        parsed_value = std::stoul(value, &parsed_characters);
    }
    catch(const std::exception&)
    {
        return false;
    }

    if(parsed_characters != value.size() || parsed_value == 0 ||
       parsed_value > std::numeric_limits<unsigned>::max())
        return false;

    result = static_cast<unsigned>(parsed_value);
    return true;
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <output_file> [-nchans <channels>] [-nacc <spectra>]"
                     " [-threads <count>]\n";
        return 1;
    }

    unsigned nchans = 512;
    unsigned nacc = 32;
    // Each worker owns a large time-domain scratch buffer, so use a conservative
    // default and let deployments opt in to more concurrency.
    unsigned number_of_threads = 2;

    for(int i = 2; i < argc; ++i)
    {
        const std::string option = argv[i];
        unsigned* destination = nullptr;
        if(option == "-nchans") destination = &nchans;
        else if(option == "-nacc") destination = &nacc;
        else if(option == "-threads") destination = &number_of_threads;

        if(!destination || i + 1 >= argc || !parse_unsigned(argv[++i], *destination))
        {
            std::cerr << "Unknown, incomplete, or invalid option: " << option << '\n';
            return 1;
        }
    }

    RcptRingBuffer<int8_t> rcpt_buf(8192, 4096, 128);
    PulsarUdpReceptor rcpt_pulsar(rcpt_buf, std::string("192.168.22.112"), 2);

    std::ofstream out_file(argv[1], std::ios::binary);
    if(!out_file)
    {
        std::cerr << "Cannot open output file: " << argv[1] << '\n';
        return 1;
    }

    try
    {
        PulsarStreamProcessor processor(nchans, nacc, number_of_threads);
        std::exception_ptr processor_error;
        std::thread receptor_thread([&] { rcpt_pulsar.run(); });
        std::thread processor_thread([&] {
            try
            {
                processor.run(rcpt_buf, out_file);
            }
            catch(...)
            {
                processor_error = std::current_exception();
                rcpt_buf.abort(true);
            }
        });
        receptor_thread.join();
        processor_thread.join();
        if(processor_error) std::rethrow_exception(processor_error);
    }
    catch(const std::exception& error)
    {
        rcpt_buf.abort(true);
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
