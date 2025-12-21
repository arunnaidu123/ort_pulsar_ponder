#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include "../LegacyDedispersion.h"
#include "../kernels/kernels.h"

namespace ort {
namespace ponder {
namespace modules {
namespace legacy_dedispersion {

#define CUDA_CHECK(call)                                                          \
do {                                                                              \
    cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                     \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                   \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__       \
                  << std::endl;                                                   \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                             \
} while (0)

static const char* cufftGetErrorString(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        default: return "Unknown CUFFT error";
    }
}

#define CUFFT_CHECK(call)                                                        \
do {                                                                             \
    cufftResult err = call;                                                      \
    if (err != CUFFT_SUCCESS) {                                                  \
        std::cerr << "cuFFT Error: " << cufftGetErrorString(err)                \
                  << " (" << err << ") at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                                  \
        std::exit(EXIT_FAILURE);                                                  \
    }                                                                            \
} while (0)

LegacyDedispersion::LegacyDedispersion(unsigned nfft, float dm)
    :   _nfft(nfft)
    ,   _dm(dm)
{

    cudaMalloc((void **)&_cufft_in, sizeof(float2)*_nfft);
    cudaMalloc((void **)&_data_in, sizeof(char)*_nfft*2); //data_in has both imaginary and real parts and its actual size will be fft_len/2
    cudaMalloc((void **)&_data_out, sizeof(float)*_nfft/2); //data_in has both imaginary and real parts and its actual size will be fft_len/2
    float2* cufft_in = reinterpret_cast<float2*>(_cufft_in);
    CUFFT_CHECK(cufftPlan1d(&_plan, _nfft, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftExecC2C(_plan, cufft_in, cufft_in, CUFFT_FORWARD));
    std::cout<<"allocated buffers \n";

}

LegacyDedispersion::~LegacyDedispersion()
{
    cufftDestroy(_plan);
    cudaFree(_cufft_in);
    cudaFree(_data_in);
}

int LegacyDedispersion::dedisperse(std::vector<char>& data_in, std::vector<float>& data_out)
{
    float2* cufft_in = reinterpret_cast<float2*>(_cufft_in);
    CUDA_CHECK(cudaMemcpy(_data_in, &_data_in[_nfft], _nfft*sizeof(char), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(&_data_in[_nfft], data_in.data(), _nfft*sizeof(char), cudaMemcpyHostToDevice));
    typecaste<<<_nfft/1024,1024>>>(cufft_in, _data_in, _nfft);
    CUFFT_CHECK(cufftExecC2C(_plan, cufft_in, cufft_in, CUFFT_FORWARD));
    complexMul<<<_nfft/1024,1024>>>(cufft_in, _nfft, _dm, 334.5, -1, 16.0);
    CUFFT_CHECK(cufftExecC2C(_plan, cufft_in, cufft_in, CUFFT_INVERSE));
    scale<<<_nfft/2048,1024>>>(cufft_in, _data_out, _nfft);
    CUDA_CHECK(cudaMemcpy(data_out.data(), _data_out, data_out.size()*sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

} // namespace legacy_dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort