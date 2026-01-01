#include <iostream>
#include <vector>
#include <cmath>
//#include <fftw3.h>
#include <fstream>
#include <chrono>
#include "../utils/FilterbankHeader.h"
#include "../modules/pfb/PolyPhaseFB.h"
#include "../modules/dedispersion/GpuCoherentDedispersion.h"

// ----------------------------------------
// Simple PFB: FIR → polyphase → FFT
// ----------------------------------------
int main(int argc, char* argv[])
{
    unsigned gpu_fft_len = 128*1024;
    int fft_len = 512;      // FFT channels
    int taps = 8;         // Taps per phase
    int filter_len = fft_len * taps;     // Total FIR length
    ort::ponder::utils::FilterbankHeader<float> fil_header;

    std::ifstream infile(argv[1], std::ios::binary);
        if (!infile) {
            std::cerr << "Cannot open file\n";
            return 1;
        }

    std::ofstream outfile(argv[3], std::ios::binary);
        if (!outfile) {
            std::cerr << "Cannot open file\n";
            return 1;
        }

    fil_header.tstart(atof(argv[2]));

    fil_header.write_header(outfile);

    std::vector<char> data_in(4*filter_len);
    std::vector<std::complex<float>> data_filtered(gpu_fft_len*fft_len/4);
    std::vector<float> data_out(gpu_fft_len*fft_len/4);
    ort::ponder::modules::pfb::PolyPhaseFB pfb(fft_len, 1.0/(4*fft_len), taps);
    ort::ponder::modules::dedispersion::GpuCoherentDedispersion dedisp(fft_len/2, gpu_fft_len, atof(argv[4]));

    auto start = std::chrono::high_resolution_clock::now();
    infile.read(reinterpret_cast<char*>(&data_in[0]), data_in.size()/2);
    unsigned iter=0;
    int flag =0;
    while(!infile.eof())
    {
        infile.read(reinterpret_cast<char*>(&data_in[data_in.size()/2]), data_in.size()/2);
        pfb.exec(data_in, data_filtered, filter_len*iter/2);
        std::copy(data_in.begin()+data_in.size()/2, data_in.end(), data_in.begin());
        iter++;
        if(iter==gpu_fft_len*fft_len/(2*filter_len))
        {
            iter=0;
            dedisp.dedisperse(data_filtered, data_out);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            start = stop;
            std::cout<<"duration in seconds: "<<duration.count()/1e6<<" s\n";
            if(flag!=0) outfile.write(reinterpret_cast<char*>(data_out.data()), data_out.size()*sizeof(float));
            flag=1;
        }
    }
    return 0;
}