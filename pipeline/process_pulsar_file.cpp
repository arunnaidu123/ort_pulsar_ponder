#include <iostream>
#include <vector>
#include <cmath>
//#include <fftw3.h>
#include <fstream>
#include <chrono>
#include <thread>
#include "../utils/FilterbankHeader.h"
#include "../modules/pfb/PolyPhaseFB.h"
#include "../modules/dedispersion/GpuCoherentDedispersion8bit.h"
#include "../producers/PulsarUdpReceptor.h"
#include "../producers/RcptRingBuffer.h"
// ----------------------------------------
// Simple PFB: FIR → polyphase → FFT
// ----------------------------------------

void filterbank_thread(RcptRingBuffer<int8_t>& rcpt_buf, std::ofstream& outfile)
{
    unsigned gpu_fft_len = 64*1024;
    unsigned nchans = 512;
    unsigned flag = 0;
    unsigned nacc = 32;
    unsigned count=0;
    ort::ponder::utils::FilterbankHeader<float> fil_header;

    std::vector<std::complex<char>> data_filtered(gpu_fft_len*nchans/2);
    std::vector<float> data_out(gpu_fft_len*nchans/2);
    std::vector<float> data_avg(nchans,0);
    ort::ponder::modules::dedispersion::GpuCoherentDedispersion8bit dedisp(nchans, gpu_fft_len, 0.0);

    while(1)
    {

        auto buffer = rcpt_buf.get_readable();
        auto start = std::chrono::high_resolution_clock::now();
        std::fill(data_avg.begin(), data_avg.end(), 0);
        count=0;
        if(flag==0)
        {
            uint32_t mjd = buffer->second->mjd_day();
            uint32_t seconds = buffer->second->mjd_seconds();
            uint64_t nanoseconds = buffer->second->mjd_nanoseconds();
            long double tstart = (long double)mjd + ((long double)seconds+(long double)nanoseconds*1e-9)/86400.0;
            fil_header.tstart(tstart);
            fil_header.tsamp(nacc*1024*10e-9);
            fil_header.write_header(outfile);
        }
        //std::copy(buffer->second->begin(), buffer->second->end(), data_filtered.begin());
        dedisp.dedisperse(reinterpret_cast<char*>(&*buffer->second->begin()), data_out);

        for(unsigned int spectra=0; spectra<32*1024; ++spectra)
        {
            for(unsigned int i=0; i<nchans; ++i)
            {
                data_avg[i] += data_out[spectra*nchans+i];
            }
            count++;
            if(count%nacc==0)
            {
                for(unsigned int i=0; i<nchans; ++i)
                {
                    data_avg[i] /= nacc;
                }
                if(flag!=0) outfile.write(reinterpret_cast<char*>(data_avg.data()), data_avg.size()*sizeof(float));
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        start = stop;
        std::cout<<"duration in seconds: "<<duration.count()/1e6<<" s\n";

        flag=1;
    }

}

void convert4bittofloat(std::vector<float>& data_out,
                        const std::vector<unsigned char>& data_in)
{

    size_t n = data_in.size();
    data_out.resize(n * 2);

    for (size_t i = 0; i < n; ++i)
    {
        unsigned char byte = data_in[i];

        // Extract high nibble
        int8_t high = (byte >> 4) & 0x0F;
        if (high & 0x08)   // sign bit
            high |= 0xF0;  // sign extend to 8-bit

        // Extract low nibble
        int8_t low = byte & 0x0F;
        if (low & 0x08)
            low |= 0xF0;

        data_out[2*i]     = static_cast<float>(high);
        data_out[2*i + 1] = static_cast<float>(low);
    }
}

int main(int argc, char* argv[])
{
    ort::ponder::utils::FilterbankHeader<float> fil_header;
    fil_header.tsamp(1024*10e-9);

    std::vector<unsigned char> data_in_4bit(512);
    std::vector<float> data_in(1024);
    std::vector<std::complex<float>> fft_buf(1024/2+1);
    std::vector<float> data_out(512);

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(
        1024,
        data_in.data(),
        reinterpret_cast<fftwf_complex*>(fft_buf.data()),
        FFTW_MEASURE
    );

    std::ifstream in_file(argv[1], std::ios::binary);
    std::ofstream out_file(argv[2], std::ios::binary);
    fil_header.write_header(out_file);
    unsigned count=0;
    while(!in_file.eof())
    {
        in_file.read(reinterpret_cast<char*>(data_in_4bit.data()), 512);
        convert4bittofloat(data_in, data_in_4bit);
        fftwf_execute(plan);
        for(unsigned int i=0; i<512;++i)
        {
            data_out[511-i] = fft_buf[i].real() * fft_buf[i].real()+ fft_buf[i].imag() * fft_buf[i].imag();
        }
        out_file.write(reinterpret_cast<char*>(data_out.data()), data_out.size()*sizeof(float));
        if(count%1000==0) std::cout<<"written spectra: "<<data_out[256]<<" "<<data_out[254]<<" "<<count<<"\n";
        count++;
    }

    return 0;
}

