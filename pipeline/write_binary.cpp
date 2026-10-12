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
#include "../producers/PulsarUdpReceptor/PulsarUdpReceptor.h"
#include "../producers/PulsarUdpReceptor/RcptRingBuffer.h"
// ----------------------------------------
// Simple PFB: FIR → polyphase → FFT
// ----------------------------------------

void write_thread(RcptRingBuffer<int8_t>& rcpt_buf, std::ofstream& outfile)
{
    unsigned gpu_fft_len = 64*1024;
    unsigned nchans = 512;
    unsigned flag = 0;
    unsigned nacc = 32;
    unsigned count=0;
    ort::ponder::utils::FilterbankHeader<float> fil_header(512);

    std::vector<std::complex<char>> data_filtered(gpu_fft_len*nchans/2);
    std::vector<float> data_out(gpu_fft_len*nchans/2);
    std::vector<float> data_avg(nchans,0);
    ort::ponder::modules::dedispersion::GpuCoherentDedispersion8bit dedisp(nchans, gpu_fft_len, 0.0);

    while(1)
    {

        auto start = std::chrono::high_resolution_clock::now();
        auto buffer = rcpt_buf.get_readable();
        if(flag==0)
        {
            uint32_t mjd = buffer->second->mjd_day();
            uint32_t seconds = buffer->second->mjd_seconds();
            uint64_t nanoseconds = buffer->second->mjd_nanoseconds();
            long double tstart = (long double)mjd + ((long double)seconds+(long double)nanoseconds*1e-9)/86400.0;
            std::cout<<tstart<<" \n";
        }
        outfile.write(reinterpret_cast<char*>(&*buffer->second->begin()), buffer->second->size()*sizeof(*buffer->second->begin()));
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        start = stop;
        std::cout<<"duration in seconds: "<<duration.count()/1e6<<" s\n";
        flag = 1;
    }

}
int main(int argc, char* argv[])
{

    RcptRingBuffer<int8_t> rcpt_buf(8192, 4096, 128);
    PulsarUdpReceptor rcpt_pulsar(rcpt_buf, std::string("192.168.22.112"), 2);

    std::ofstream outfile(argv[1], std::ios::binary);
    if (!outfile) {
        std::cerr << "Cannot open file\n";
        return 1;
    }

    std::thread rcpt_thread([&]{rcpt_pulsar.run();});
    std::thread stream_thread(write_thread, std::ref(rcpt_buf), std::ref(outfile));
    rcpt_thread.join();




//    unsigned gpu_fft_len = 64*1024;
//    unsigned nchans = 512;
//    ort::ponder::utils::FilterbankHeader<float> fil_header;
//
//
//
//    //get that info from the first buffer
//    fil_header.tstart(atof(argv[2]));
//
//    fil_header.write_header(outfile);
//
//    std::vector<std::complex<float>> data_filtered(gpu_fft_len*nchans/2);
//    std::vector<float> data_out(gpu_fft_len*nchans/2);
//    ort::ponder::modules::dedispersion::GpuCoherentDedispersion dedisp(nchans, gpu_fft_len, atof(argv[4]));
//
//    auto start = std::chrono::high_resolution_clock::now();
//    if(iter==gpu_fft_len*fft_len/(2*filter_len))
//    {
//        iter=0;
//        dedisp.dedisperse(data_filtered, data_out);
//        auto stop = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//        start = stop;
//        std::cout<<"duration in seconds: "<<duration.count()/1e6<<" s\n";
//        if(flag!=0) outfile.write(reinterpret_cast<char*>(data_out.data()), data_out.size()*sizeof(float));
//        flag=1;
//    }

    return 0;
}
