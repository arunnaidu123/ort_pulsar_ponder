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
double ts =31.25e-9;
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
        double t0 = ts*((64000.0*std::floor(current_sample/64000.0))+32000.0);
        double temp = dist(gen)*(1+gaussian_pulse(t, t0, width));
        if(temp > 127) data_in[i] = 127;
        else if(temp < -127) data_in[i] = -127;
        else data_in[i] = (char)(int)(temp);
        current_sample++;
    }
}

int main(int argc, char* argv[])
{
    current_sample=0;
        unsigned nfft = 128*1024*1024;
    std::vector<char> data_in(nfft/2);
    std::vector<char> data_out(nfft);
    ort::ponder::modules::dispersed_pulsar::DispersedPulsar disp_pulsar(nfft, atof(argv[2]));

    std::ofstream outfile(argv[1], std::ios::binary);
    if (!outfile) {
        std::cerr << "Cannot open file\n";
        return 1;
    }

    unsigned flag =0;
    for(unsigned int i=0; i<15; i++)
    {
        sim_pulsar(data_in);
        disp_pulsar.disperse(data_in, data_out);
        if (i!=0) outfile.write(reinterpret_cast<char*>(data_out.data()), data_out.size());
        std::cout<<i<<" time: "<<current_sample*ts<<" s \n";
    }
    return 0;
}