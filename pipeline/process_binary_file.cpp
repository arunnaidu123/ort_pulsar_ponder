#include <fftw3.h>
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <cstdlib>
#include <chrono>
#include "../utils/FilterbankHeader.h"
#include "../modules/pfb/PolyPhaseFB.h"
#include "SsePower.h"
constexpr double PI = 3.14159265358979323846;

class ProcessBlock
{
    public:
        ProcessBlock(unsigned buffer_size, unsigned nchans, unsigned nacc)
        : _buffer_size(buffer_size)
        , _data_in(buffer_size)
        , _data_tmp(buffer_size, 0.0)
        , _data_out(buffer_size)
        , _time_buf(1024)
        , _fft_buf(1024)
        , _fft_buf_nchans(nchans+1)
        , _power_buf(nchans)
        {
            _plan = fftwf_plan_dft_1d(1024, reinterpret_cast<fftwf_complex*>(_fft_buf.data()), reinterpret_cast<fftwf_complex*>(_time_buf.data()), FFTW_BACKWARD, FFTW_MEASURE);
            _plan_nchans = fftwf_plan_dft_r2c_1d(2*nchans, _data_tmp.data(), reinterpret_cast<fftwf_complex*>(_fft_buf_nchans.data()), FFTW_MEASURE);
        }

        ~ProcessBlock()
        {
            fftwf_destroy_plan(_plan);
            fftwf_destroy_plan(_plan_nchans);
        }

        void process_file_incoherent(std::ifstream& in_file, std::ofstream& out_file, unsigned nchans, unsigned nacc);

    private:
        fftwf_plan _plan;
        fftwf_plan _plan_nchans;
        std::size_t _buffer_size;
        std::vector<int8_t> _data_in;
        std::vector<float> _data_tmp;
        std::vector<std::complex<float>> _data_out;
        std::vector<std::complex<float>> _time_buf;
        std::vector<std::complex<float>> _fft_buf;
        std::vector<std::complex<float>> _fft_buf_nchans;
        std::vector<float> _power_buf;
};

void ProcessBlock::process_file_incoherent(std::ifstream& in_file, std::ofstream& out_file, unsigned nchans, unsigned nacc)
{
    std::vector<float> output_buffer;
    while(in_file.read(reinterpret_cast<char*>(_data_in.data()), _buffer_size) || in_file.gcount() != 0)
    {
        auto start = std::chrono::high_resolution_clock::now();
        const auto bytes_read = static_cast<std::size_t>(in_file.gcount());
        unsigned number_of_spectra = bytes_read/1024;

        for(unsigned int i=0; i<number_of_spectra; ++i)
        {
            for(unsigned channel=0; channel<512; ++channel)
            {
                const std::complex<float> value(
                    _data_in[i*1024+2*channel],
                    _data_in[i*1024+2*channel+1]);
                _fft_buf[512+channel] = value;
                _fft_buf[511-channel] = std::conj(value);
            }
            fftwf_execute(_plan);
            for(unsigned int s=0; s<1024; ++s)
            {
                _data_tmp[i*1024 + s] = _time_buf[s].real();
            }
        }

        number_of_spectra = (number_of_spectra*1024)/(2*nchans);
        const unsigned output_spectra = number_of_spectra/nacc;
        output_buffer.resize(static_cast<std::size_t>(output_spectra)*nchans);
        for(unsigned group = 0; group<output_spectra; ++group)
        {
            std::fill(_power_buf.begin(), _power_buf.end(), 0.0);
            for(unsigned n = 0; n<nacc; ++n)
            {
                const unsigned spectra = group*nacc+n;
                fftwf_execute_dft_r2c(
                    _plan_nchans,
                    _data_tmp.data() + static_cast<std::size_t>(2)*nchans*spectra,
                    reinterpret_cast<fftwf_complex*>(_fft_buf_nchans.data()));

                accumulate_power_sse4(_fft_buf_nchans.data(), _power_buf.data(), nchans);
            }
            normalize_power_sse4(_power_buf.data(), nchans, 1.0f/nacc);
            std::copy(_power_buf.begin(), _power_buf.end(),
                      output_buffer.begin() + static_cast<std::size_t>(group)*nchans);
        }
        out_file.write(reinterpret_cast<const char*>(output_buffer.data()),
                       output_buffer.size()*sizeof(float));

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        start = stop;
        std::cout<<"duration in seconds: "<<duration.count()/1e6<<" s\n";
    }
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> -o <out_file> -nacc <nacc> -nchans <nchans>\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string out_file_name;
    unsigned nchans = 2048;
    unsigned nacc = 32;

    for(int i=2; i<argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "-o" && i+1 < argc) out_file_name = argv[++i];
        else if(arg == "-nacc" && i+1 < argc) nacc = std::atoi(argv[++i]);
        else if(arg == "-nchans" && i+1 < argc) nchans = std::atoi(argv[++i]);
    }

    std::ifstream in_file(input_file, std::ios::binary);
    std::ofstream out_file(out_file_name, std::ios::binary);

    ProcessBlock process_binary(32*1024*1024, nchans, nacc);
    ort::ponder::utils::FilterbankHeader<float> fil_header(nchans);
    fil_header.tstart(61236.2123456);
    fil_header.tsamp(nacc*nchans*2*10e-9);
    fil_header.write_header(out_file);

    process_binary.process_file_incoherent(in_file, out_file, nchans, nacc);

    return 0;
}
