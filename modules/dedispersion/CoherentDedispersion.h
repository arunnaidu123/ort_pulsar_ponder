#ifndef ORT_PONDER_MODULES_DEDISPERSION_COHERENTDEDISPERSION_H
#define ORT_PONDER_MODULES_DEDISPERSION_COHERENTDEDISPERSION_H


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
namespace dedispersion {

class CoherentDedispersion
{

public:
    CoherentDedispersion() = delete;
    CoherentDedispersion(unsigned _nchans, unsigned fft_len, float dm);
    ~CoherentDedispersion();

    int calPhase(float dm);

    int dedisperse(std::vector<std::complex<float>>& data_in, std::vector<float>& data_out);

private:
    unsigned _nchans;
    unsigned _gpu_fft_len;
    float _dm;
    char* _data_in;
    double* _phase;
    float* _data_out;
    void* _cufft_in;
    void* _cufft_out;
#ifdef __CUDACC__
    cufftHandle _plan;
#endif //__CUDACC__
};

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort


#endif //ORT_PONDER_MODULES_DEDISPERSION_COHERENTDEDISPERSION_H