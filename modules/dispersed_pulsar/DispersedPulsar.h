#ifndef ORT_PONDER_MODULES_DISPERSEDPULSAR_DISPERSEDPULSAR_H
#define ORT_PONDER_MODULES_DISPERSEDPULSAR_DISPERSEDPULSAR_H


#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#endif //__CUDACC__
//#include <helper_functions.h>

//#include <helper_cuda.h>

namespace ort {
namespace ponder {
namespace modules {
namespace dispersed_pulsar {

class DispersedPulsar
{

public:
    DispersedPulsar() = delete;
    DispersedPulsar(unsigned nfft, float dm);
    ~DispersedPulsar();

    int disperse(std::vector<char>& data_in, std::vector<char>& data_out);

private:
    unsigned _nfft;
    float _dm;
    char* _data_in;
    char* _data_out;
    void* _cufft_in;
    int _plan;
};

} // namespace dispersed_pulsar
} // namespace modules
} // namespace ponder
} // namespace ort


#endif //ORT_PONDER_MODULES_DISPERSEDPULSAR_DISPERSEDPULSAR_H