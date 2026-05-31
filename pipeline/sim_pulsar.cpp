#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <fstream>
#include <chrono>
#include <random>
#include "../utils/FilterbankHeader.h"
#include "../modules/pfb/PolyPhaseFB.h"
#include "../modules/dispersed_pulsar/DispersedPulsar.h"

unsigned long  current_sample;
std::random_device rd;               // non-deterministic seed
std::mt19937 gen(rd());              // Mersenne Twister RNG
std::normal_distribution<> dist(0.0, 16.0);   // mean=0, stddev=1
double ts =10e-9;
double period = 2.0e-3;
double width = 2.0e-5;

double gaussian_pulse(double t, double t0, double sigma, double A = 2.5)
{
    double x = (t - t0) / sigma;
    return A * std::exp(-0.5 * x * x);
}

void sim_pulsar(std::vector<char>& data_in)
{
    for(unsigned int i=0; i<data_in.size(); ++i)
    {
        double t = current_sample*ts;
        double t0 = ts*((200000.0*std::floor(current_sample/200000.0))+100000.0);
        double temp = dist(gen)*(1+gaussian_pulse(t, t0, width));
        if(temp > 127) data_in[i] = 127;
        else if(temp < -127) data_in[i] = -127;
        else data_in[i] = (char)(int)(temp);
        current_sample++;
    }
}

void convert8bitto4bit(const std::vector<char>& data_in,
                       std::vector<unsigned char>& data_out)
{
    size_t n = data_in.size();
    size_t out_size = (n + 1) / 2;

    data_out.resize(out_size);

    for (size_t i = 0; i < n; i ++)
    {
        // ---- First sample ----
        int v0 = static_cast<int>(data_in[i]);

        // Scale: divide by 16 (arithmetic shift keeps sign)
        v0 >>= 4;

        // Saturate to [-8, 7]
        v0 = std::max(-8, std::min(7, v0));

        // Convert to unsigned nibble (two's complement 4-bit)
        unsigned char nibble0 = static_cast<unsigned char>(v0 & 0x0F);

//        // ---- Second sample (if exists) ----
//        unsigned char nibble1 = 0;
//        if (i + 1 < n)
//        {
//            int v1 = static_cast<int>(data_in[i + 1]);
//            v1 >>= 4;
//            v1 = std::max(-8, std::min(7, v1));
//            nibble1 = static_cast<unsigned char>(v1 & 0x0F);
//        }
//
        // Pack: first sample in high nibble
        data_out[i] = nibble0;
    }
}

int main(int argc, char* argv[])
{
    current_sample=0;
        unsigned nfft = 128*1024*1024;
    std::vector<char> data_in(nfft/2);
    std::vector<char> data_out(nfft/2);
    std::vector<unsigned char> data_out_4bit(nfft/2);
    ort::ponder::modules::dispersed_pulsar::DispersedPulsar disp_pulsar(nfft, atof(argv[2]));

    std::ofstream outfile(argv[1], std::ios::binary);
    if (!outfile) {
        std::cerr << "Cannot open file\n";
        return 1;
    }

    unsigned flag =0;
    for(unsigned int i=0; i<100; i++)
    {
        sim_pulsar(data_in);
        disp_pulsar.disperse(data_in, data_out);
        convert8bitto4bit(data_out, data_out_4bit);
        if (i!=0) outfile.write(reinterpret_cast<char*>(data_out_4bit.data()), data_out_4bit.size());
        std::cout<<i<<" time: "<<current_sample*ts<<" s \n";
    }
    return 0;
}