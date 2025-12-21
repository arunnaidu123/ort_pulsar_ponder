#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <fstream>
#include <chrono>
#include <random>
#include "../utils/DedispHeader.h"
#include "../modules/legacy_dedispersion/LegacyDedispersion.h"

//#include "../modules/pfb/PolyPhaseFB.h"
//#include "../modules/dispersed_pulsar/DispersedPulsar.h"


int main(int argc, char* argv[])
{
    unsigned nfft = 128*1024*1024;
    std::vector<char> data_in(nfft);
    std::vector<float> data_out(nfft/2);
    ort::ponder::modules::legacy_dedispersion::LegacyDedispersion dedisp_data(nfft, atof(argv[2]));
    ort::ponder::utils::DedispHeader dedispersion_header(atof(argv[2]));

    std::ifstream infile(argv[1], std::ios::binary);
    if (!infile)
    {
        std::cerr << "Cannot open input file\n";
        return 1;
    }

    std::ofstream outfile(argv[3], std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Cannot open output file\n";
        return 1;
    }
    dedispersion_header.write_header(outfile);

    unsigned flag =0;
    while(!infile.eof())
    {
        infile.read(data_in.data(), sizeof(char)*nfft);
        dedisp_data.dedisperse(data_in, data_out);
        if (flag!=0) outfile.write(reinterpret_cast<char*>(data_out.data()), data_out.size()*sizeof(float));
        else flag =1;
    }
    return 0;
}