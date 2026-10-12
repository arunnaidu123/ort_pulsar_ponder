#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <thread>
#include "../utils/FilterbankHeader.h"
#include "../modules/pfb/PolyPhaseFB.h"
#include "../producers/IpsUdpReceptor/IpsUdpReceptor.h"
#include "../producers/IpsUdpReceptor/RcptRingBuffer.h"
// ----------------------------------------
// Simple PFB: FIR → polyphase → FFT
// ----------------------------------------

int main(int argc, char* argv[])
{

    RcptRingBuffer<int8_t> rcpt_buf(10, 4096, 128);
    IpsUdpReceptor rcpt_ips(rcpt_buf, std::string("192.168.23.113"), 2);

    unsigned number_of_beams = argc-1;
    std::cout<<"number of beams: "<<number_of_beams<<"\n";
    std::vector<std::ofstream> outfiles;

    for(unsigned int i=0; i<number_of_beams; ++i)
    {
        outfiles.push_back(std::ofstream(argv[1+i], std::ios::binary));
        if(!outfiles[i])
        {
            std::cerr << "Cannot open file\n";
            return 1;
        }
    }

    std::thread rcpt_thread([&]{rcpt_ips.run();});
    ort::ponder::utils::FilterbankHeader<float> fil_header(512);
    std::vector<float> temp(512);
    unsigned flag=0;
    unsigned long count=0;
    while(1)
    {
        auto buffer = rcpt_buf.get_readable();
        auto start = std::chrono::high_resolution_clock::now();
        if(flag==0)
        {
            uint32_t mjd = buffer->second->mjd_day();
            uint32_t seconds = buffer->second->mjd_seconds();
            uint64_t nanoseconds = buffer->second->mjd_nanoseconds();
            long double tstart = (long double)mjd + ((long double)seconds+(long double)nanoseconds*1e-9)/86400.0;
            fil_header.tstart(tstart);
            fil_header.tsamp(1024*1024*10e-9);
            for(unsigned i=0; i<number_of_beams; ++i) fil_header.write_header(outfiles[i]);
            flag=1;
        }

        for(unsigned int beam=0; beam<number_of_beams; ++beam)
        {
            auto buf = reinterpret_cast<float*>(&*buffer->second->begin());
            for(unsigned int channel=0; channel<512; ++channel)
            {
                temp[512-channel-1] = buf[512*beam+channel];
            }
            temp[511] = 0.0;
            outfiles[beam].write(reinterpret_cast<char*>(&*temp.begin()), 512*sizeof(float));
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        start = stop;
        count++;
        if(count%100==0) std::cout<<"duration in seconds: "<<duration.count()/1e6<<" s\n";
    }

    rcpt_thread.join();
    return 0;
}
