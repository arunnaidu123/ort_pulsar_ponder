#include <fftw3.h>
#include <vector>
#include <complex>
#include <cmath>
#include "../utils/FilterbankHeader.h"
constexpr double PI = 3.14159265358979323846;

class ProcessBlock
{
    public:
        ProcessBlock()
        : _ant1(33*1024)
        , _ant2(33*1024)
        , _data_in(33*1024)
        , _time_buf(1024)
        , _fft_buf(1024)
        , _temp_buf(1024)
        , _power_buf(512)
        , _frac_delay(512)
        {
            _plan = fftwf_plan_dft_1d(1024, reinterpret_cast<fftwf_complex*>(_time_buf.data()), reinterpret_cast<fftwf_complex*>(_fft_buf.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        }

        ~ProcessBlock()
        {
            fftwf_destroy_plan(_plan);
        }

        void process_file_frac(std::ifstream& in_file, std::ofstream& out_file, double shift);
        void process_file_incohernet(std::ifstream& in_file, std::ofstream& out_file, int shift);
        void process_file_int(std::ifstream& in_file, std::ofstream& out_file, int shift);

    private:
        fftwf_plan _plan;
        std::vector<std::complex<float>> _ant1;
        std::vector<std::complex<float>> _ant2;
        std::vector<uint8_t> _data_in;
        std::vector<std::complex<float>> _time_buf;
        std::vector<std::complex<float>> _fft_buf;
        std::vector<std::complex<float>> _temp_buf;
        std::vector<float> _power_buf;
        std::vector<std::complex<float>> _frac_delay;
};

inline int8_t unpack4bit(uint8_t nibble)
{
    // nibble must be 0–15
    return (int8_t)(nibble << 4) >> 4;
}

void ProcessBlock::process_file_int(std::ifstream& in_file, std::ofstream& out_file, int shift)
{
    unsigned count=0;
    while(!in_file.eof() && count < 915528)
    {
        if(count%100==0) std::cout<<"writing "<<count<<"\n";
        count++;
        in_file.read(reinterpret_cast<char*>(&*(_data_in.begin()+1024)), 32*1024);

        for(unsigned int i=0; i<33*1024; ++i)
        {
            uint8_t byte = _data_in[i];

            int8_t v1 = unpack4bit(byte & 0x0F);        // lower nibble
            int8_t v2 = unpack4bit((byte >> 4) & 0x0F); // upper nibble

            _ant1[i] = std::complex<float>((float)v1, 0.0f);
            _ant2[i] = std::complex<float>((float)v2, 0.0f);
        }

        std::copy(_data_in.begin()+32*1024, _data_in.end(), _data_in.begin());
        std::fill(_power_buf.begin(), _power_buf.end(), 0);
        for(unsigned int k=0; k<32; ++k)
        {

            auto ant1_it = _ant1.begin()+k*1024+512;
            auto ant2_it = _ant2.begin()+k*1024+512+shift;
            std::copy(ant1_it, ant1_it+1024, _time_buf.begin());
            fftwf_execute(_plan);
            for(unsigned int s=0; s<1024; ++s)
            {
                _temp_buf[s] = _fft_buf[s];
            }
            std::copy(ant2_it, ant2_it+1024, _time_buf.begin());
            fftwf_execute(_plan);
            for(unsigned int s=0; s<1024; ++s)
            {
                _temp_buf[s] = std::complex<float>(_temp_buf[s].real()+_fft_buf[s].real(), _temp_buf[s].imag()+_fft_buf[s].imag());
            }
            for(unsigned int s=0; s<512; ++s)
            {
                _power_buf[s] += std::abs(_temp_buf[511-s]);
            }
        }
        for(unsigned int s=0; s<512; ++s) _power_buf[s] /= 32.0;
        out_file.write(reinterpret_cast<char*>(_power_buf.data()), _power_buf.size()*sizeof(float));

    }
}

void ProcessBlock::process_file_incohernet(std::ifstream& in_file, std::ofstream& out_file, int shift)
{
    unsigned count=0;
    while(!in_file.eof() && count < 915528)
    {
        if(count%100==0) std::cout<<"writing "<<count<<"\n";
        count++;
        in_file.read(reinterpret_cast<char*>(&*(_data_in.begin()+1024)), 32*1024);

        for(unsigned int i=0; i<33*1024; ++i)
        {
            uint8_t byte = _data_in[i];

            int8_t v1 = unpack4bit(byte & 0x0F);        // lower nibble
            int8_t v2 = unpack4bit((byte >> 4) & 0x0F); // upper nibble

            _ant1[i] = std::complex<float>((float)v1, 0.0f);
            _ant2[i] = std::complex<float>((float)v2, 0.0f);
        }

        std::copy(_data_in.begin()+32*1024, _data_in.end(), _data_in.begin());
        std::fill(_power_buf.begin(), _power_buf.end(), 0);
        for(unsigned int k=0; k<32; ++k)
        {

            auto ant1_it = _ant1.begin()+k*1024+512;
            auto ant2_it = _ant2.begin()+k*1024+512+shift;
            std::copy(ant1_it, ant1_it+1024, _time_buf.begin());
            fftwf_execute(_plan);
            for(unsigned int s=0; s<1024; ++s)
            {
                _temp_buf[s] = _fft_buf[s];
            }
            std::copy(ant2_it, ant2_it+1024, _time_buf.begin());
            fftwf_execute(_plan);
            for(unsigned int s=0; s<1024; ++s)
            {
                _temp_buf[s] = std::complex<float>(_temp_buf[s].real()+_fft_buf[s].real(), _temp_buf[s].imag()+_fft_buf[s].imag());
            }
            for(unsigned int s=0; s<512; ++s)
            {
                _power_buf[s] += std::abs(_temp_buf[511-s]);
            }
        }
        for(unsigned int s=0; s<512; ++s) _power_buf[s] /= 32.0;
        out_file.write(reinterpret_cast<char*>(_power_buf.data()), _power_buf.size()*sizeof(float));

    }
}

void ProcessBlock::process_file_frac(std::ifstream& in_file, std::ofstream& out_file, double shift)
{
    int shift_int = (int) shift;
    double fraction = shift - (double)shift_int;
    fraction *= 10.0;
    double tau = fraction*1e-9;

    for(unsigned int i=0; i<512; ++i)
    {
        double frequency = i*(50.0/512);
        frequency *= 1e6;
        double phi = 2.0 * PI * frequency * tau;
        _frac_delay[i].real(std::cos(phi));
        _frac_delay[i].imag(std::sin(phi));
    }

    unsigned count=0;
    while(!in_file.eof() && count < 915528)
    {
        if(count%100==0) std::cout<<"writing "<<count<<"\n";
        count++;
        in_file.read(reinterpret_cast<char*>(&*(_data_in.begin()+1024)), 32*1024);

        for(unsigned int i=0; i<33*1024; ++i)
        {
            uint8_t byte = _data_in[i];

            int8_t v1 = unpack4bit(byte & 0x0F);        // lower nibble
            int8_t v2 = unpack4bit((byte >> 4) & 0x0F); // upper nibble

            _ant1[i] = std::complex<float>((float)v1, 0.0f);
            _ant2[i] = std::complex<float>((float)v2, 0.0f);
        }

        std::copy(_data_in.begin()+32*1024, _data_in.end(), _data_in.begin());
        std::fill(_power_buf.begin(), _power_buf.end(), 0);
        for(unsigned int k=0; k<32; ++k)
        {

            auto ant1_it = _ant1.begin()+k*1024+512;
            auto ant2_it = _ant2.begin()+k*1024+512+shift_int;
            std::copy(ant1_it, ant1_it+1024, _time_buf.begin());
            fftwf_execute(_plan);
            for(unsigned int s=0; s<1024; ++s)
            {
                _temp_buf[s] = _fft_buf[s];
            }
            std::copy(ant2_it, ant2_it+1024, _time_buf.begin());
            fftwf_execute(_plan);
            for(unsigned int s=0; s<512; ++s)
            {
                _fft_buf[s] *= _frac_delay[s];
                _temp_buf[s] = std::complex<float>(_temp_buf[s].real()+_fft_buf[s].real(), _temp_buf[s].imag()+_fft_buf[s].imag());
            }
            for(unsigned int s=0; s<512; ++s)
            {
                _power_buf[s] += std::abs(_temp_buf[511-s]);
            }
        }
        for(unsigned int s=0; s<512; ++s) _power_buf[s] /= 32.0;
        out_file.write(reinterpret_cast<char*>(_power_buf.data()), _power_buf.size()*sizeof(float));

    }
}

int main(int argc, char* argv[])
{
    ort::ponder::utils::FilterbankHeader<float> fil_header;
    float shift = atof(argv[3]);
    ProcessBlock process_binary;
    std::ifstream in_file(argv[1], std::ios::binary);
    std::ofstream out_file(argv[2], std::ios::binary);

    fil_header.tstart(61094.702110557038);
    fil_header.tsamp(32*1024*10e-9);
    fil_header.write_header(out_file);

    process_binary.process_file_frac(in_file, out_file, shift);

    return 0;
}