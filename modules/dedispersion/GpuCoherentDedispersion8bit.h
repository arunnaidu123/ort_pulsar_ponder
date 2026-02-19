#ifndef ORT_PONDER_MODULES_DEDISPERSION_GPUCOHERENTDEDISPERSION8BIT_H
#define ORT_PONDER_MODULES_DEDISPERSION_GPUCOHERENTDEDISPERSION8BIT_H


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

class GpuCoherentDedispersion8bit
{

public:
    GpuCoherentDedispersion8bit() = delete;
    GpuCoherentDedispersion8bit(int _nchans, int fft_len, float dm);
    ~GpuCoherentDedispersion8bit();

    int calPhase();
    int dedisperse(char* data_in, std::vector<float>& data_out);

private:
    int _nchans;
    int _gpu_fft_len;
    float _dm;
    double* _phase;
    float* _gpu_data_out;
    float2* _gpu_in;
    float2* _gpu_mid;
    float2* _gpu_out;
    int8_t* _gpu_data_in;
    std::vector<float> _cpu_temp;
    cufftHandle _planf;
    cufftHandle _plani;
};

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort


#endif //ORT_PONDER_MODULES_DEDISPERSION_GPUCOHERENTDEDISPERSION8BIT_H