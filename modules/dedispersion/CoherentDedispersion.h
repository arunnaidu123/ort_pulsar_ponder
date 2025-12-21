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
#include <fftw3.h>
#include <algorithm>
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

    int calPhase();



    int dedisperse(std::vector<std::complex<float>>& data_in, std::vector<float>& data_out);

private:
    unsigned _nchans;
    unsigned _gpu_fft_len;
    float _dm;
    std::vector<std::complex<float>> _phase;
    std::vector<double> _temp_phase;
    std::vector<fftwf_complex> _cpu_in;
    std::vector<fftwf_complex> _cpu_mid;
    std::vector<fftwf_complex> _cpu_out;
    fftwf_plan _cpu_fwd;
    fftwf_plan _cpu_inv;
#ifdef __CUDACC__
    cufftHandle _planf;
    cufftHandle _plani;
#endif //__CUDACC__
};

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort


#endif //ORT_PONDER_MODULES_DEDISPERSION_COHERENTDEDISPERSION_H