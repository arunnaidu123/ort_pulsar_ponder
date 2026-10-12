#include <chrono>
#include <complex>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../modules/gpu_pulsar_processing/GpuPulsarProcessing.h"
#include "../producers/PulsarUdpReceptor/PulsarUdpReceptor.h"
#include "../producers/PulsarUdpReceptor/RcptRingBuffer.h"
#include "../utils/FilterbankHeader.h"

namespace
{

using ort::ponder::modules::GpuPulsarProcessing;

void process_stream(RcptRingBuffer<int8_t>& ring_buffer,
                    std::ofstream& output_file,
                    unsigned nchans,
                    unsigned nacc,
                    float dm,
                    int device)
{
    GpuPulsarProcessing processor(nchans, nacc, dm, device);
    ort::ponder::utils::FilterbankHeader<unsigned short> header(nchans);
    std::vector<unsigned short> output;
    bool header_written = false;

    while(!ring_buffer.abort())
    {
        std::shared_ptr<RcptRingBuffer<int8_t>::BufferType> buffer;
        try
        {
            buffer = ring_buffer.get_readable();
        }
        catch(const std::runtime_error&)
        {
            if(ring_buffer.abort()) break;
            throw;
        }

        if(!header_written)
        {
            const auto& data = *buffer->second;
            const long double tstart =
                static_cast<long double>(data.mjd_day()) +
                (static_cast<long double>(data.mjd_seconds()) +
                 static_cast<long double>(data.mjd_nanoseconds()) * 1e-9L) /
                    86400.0L;
            header.tstart(tstart);
            header.tsamp(nacc * 2.0 * nchans * 1e-8);
            header.fch1(346.5 - (50.0/(double)(2*nchans)));
            header.write_header(output_file);

            header_written = true;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        processor.process(*buffer->second, output);
        output_file.write(reinterpret_cast<const char*>(output.data()),
                          output.size() * sizeof(unsigned short));
        output_file.flush();
        if(!output_file) throw std::runtime_error("error writing filterbank output");

        const std::chrono::duration<double> elapsed =
            std::chrono::high_resolution_clock::now() - start;
        std::cout << "GPU " << processor.device() << " processed one buffer in "
                  << elapsed.count() << " s\n";
    }
}

bool parse_unsigned(const std::string& value, unsigned& result)
{
    std::size_t parsed = 0;
    unsigned long number = 0;
    try { number = std::stoul(value, &parsed); }
    catch(const std::exception&) { return false; }
    if(parsed != value.size() || number == 0 ||
       number > std::numeric_limits<unsigned>::max())
        return false;
    result = static_cast<unsigned>(number);
    return true;
}

bool parse_float(const std::string& value, float& result)
{
    std::size_t parsed = 0;
    float number = 0.0;
    try { number = std::stof(value, &parsed); }
    catch(const std::exception&) { return false; }
    if(parsed != value.size() || number > std::numeric_limits<float>::max())
        return false;
    result = static_cast<float>(number);
    return true;
}

bool parse_device(const std::string& value, int& result)
{
    std::size_t parsed = 0;
    long number = 0;
    try { number = std::stol(value, &parsed); }
    catch(const std::exception&) { return false; }
    if(parsed != value.size() || number < 0 ||
       number > std::numeric_limits<int>::max())
        return false;
    result = static_cast<int>(number);
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    static_assert(sizeof(std::complex<int8_t>) == 2,
                  "GPU input transfer requires packed complex int8 samples");
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <output_file> [-nchans <channels>] [-nacc <spectra>]"
                     " [-device <index>]\n";
        return 1;
    }

    unsigned nchans = 2048;
    unsigned nacc = 32;
    float dm = 0.0;
    int device = 0;
    for(int i = 2; i < argc; ++i)
    {
        const std::string option = argv[i];
        if(i + 1 >= argc)
        {
            std::cerr << "Incomplete option: " << option << '\n';
            return 1;
        }
        const std::string value = argv[++i];
        const bool valid =
            (option == "-nchans" && parse_unsigned(value, nchans)) ||
            (option == "-nacc" && parse_unsigned(value, nacc)) ||
            (option == "-dm" && parse_float(value, dm)) ||
            (option == "-device" && parse_device(value, device));
        if(!valid)
        {
            std::cerr << "Unknown or invalid option: " << option << '\n';
            return 1;
        }
    }

    RcptRingBuffer<int8_t> ring_buffer(8192, 4096, 128);
    PulsarUdpReceptor receptor(ring_buffer, std::string("192.168.22.112"), 2);
    std::ofstream output_file(argv[1], std::ios::binary);
    if(!output_file)
    {
        std::cerr << "Cannot open output file: " << argv[1] << '\n';
        return 1;
    }

    std::exception_ptr receptor_error;
    std::exception_ptr processor_error;
    std::thread receptor_thread([&] {
        try { receptor.run(); }
        catch(...)
        {
            receptor_error = std::current_exception();
            ring_buffer.abort(true);
        }
    });
    std::thread processor_thread([&] {
        try { process_stream(ring_buffer, output_file, nchans, nacc, dm, device); }
        catch(...)
        {
            processor_error = std::current_exception();
            ring_buffer.abort(true);
        }
    });
    receptor_thread.join();
    processor_thread.join();
    try
    {
        if(receptor_error) std::rethrow_exception(receptor_error);
        if(processor_error) std::rethrow_exception(processor_error);
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
