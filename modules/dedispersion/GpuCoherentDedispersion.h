#ifndef ORT_PONDER_MODULES_DEDISPERSION_GPUCOHERENTDEDISPERSION_H
#define ORT_PONDER_MODULES_DEDISPERSION_GPUCOHERENTDEDISPERSION_H


#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
//#include <cufftw.h>
#include <algorithm>
//#include <helper_functions.h>

//#include <helper_cuda.h>

namespace ort {
namespace ponder {
namespace modules {
namespace dedispersion {

class GpuCoherentDedispersion
{

public:
    GpuCoherentDedispersion() = delete;
    GpuCoherentDedispersion(int _nchans, int fft_len, float dm);
    ~GpuCoherentDedispersion();

    int calPhase();
    int dedisperse(std::vector<std::complex<float>>& data_in, std::vector<float>& data_out);

private:
    int _nchans;
    int _gpu_fft_len;
    float _dm;
    double* _phase;
    float* _gpu_data_out;
    float2* _gpu_in;
    float2* _gpu_mid;
    float2* _gpu_out;
    std::vector<float> _cpu_temp;
    cufftHandle _planf;
    cufftHandle _plani;
};

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort


#endif //ORT_PONDER_MODULES_DEDISPERSION_GPUCOHERENTDEDISPERSION_H