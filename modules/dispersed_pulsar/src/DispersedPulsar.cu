#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include "../DispersedPulsar.h"
#include "../kernels/kernels.h"

namespace ort {
namespace ponder {
namespace modules {
namespace dispersed_pulsar {

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

DispersedPulsar::DispersedPulsar(unsigned nfft, float dm)
    :   _nfft(nfft)
    ,   _dm(dm)
    ,   _num_spectra((nfft/2)/_channel_fft_len)
{

    cudaMalloc((void **)&_cufft_in, sizeof(float2)*_nfft);
    cudaMalloc((void **)&_data_in, sizeof(char)*_nfft); //data_in has both imaginary and real parts and its actual size will be fft_len/2
    cudaMalloc((void **)&_data_out, sizeof(char)*_nfft); //data_in has both imaginary and real parts and its actual size will be fft_len/2
    float2* cufft_in = reinterpret_cast<float2*>(_cufft_in);
    CUFFT_CHECK(cufftPlan1d(&_plan, _nfft, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftExecC2C(_plan, cufft_in, cufft_in, CUFFT_FORWARD));

    _real_time_series.resize(_nfft/2);
    _channelizer_time_buffer = static_cast<float*>(fftwf_malloc(sizeof(float) * (_nfft/2)));
    _channelizer_freq_buffer = fftwf_malloc(sizeof(fftwf_complex) * _num_spectra * (_channel_fft_len/2 + 1));

    int channel_fft_len = static_cast<int>(_channel_fft_len);
    fftwf_plan channelizer_plan = fftwf_plan_many_dft_r2c(1, &channel_fft_len, static_cast<int>(_num_spectra)
                                                        , _channelizer_time_buffer, nullptr, 1, channel_fft_len
                                                        , static_cast<fftwf_complex*>(_channelizer_freq_buffer), nullptr, 1
                                                        , channel_fft_len/2 + 1
                                                        , FFTW_MEASURE
                                                        );
    _channelizer_plan = channelizer_plan;
    std::cout<<"allocated buffers \n";

}

DispersedPulsar::~DispersedPulsar()
{
    fftwf_destroy_plan(static_cast<fftwf_plan>(_channelizer_plan));
    fftwf_free(_channelizer_time_buffer);
    fftwf_free(_channelizer_freq_buffer);
    cufftDestroy(_plan);
    cudaFree(_cufft_in);
    cudaFree(_data_in);
}

int DispersedPulsar::disperse(std::vector<char>& data_in, std::vector<int8_t>& data_out)
{
    float2* cufft_in = reinterpret_cast<float2*>(_cufft_in);
    CUDA_CHECK(cudaMemcpy(_data_in, &_data_in[_nfft/2], _nfft*sizeof(char)/2, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(&_data_in[_nfft/2], data_in.data(), _nfft*sizeof(char)/2, cudaMemcpyHostToDevice));
    typecaste1<<<_nfft/1024,1024>>>(cufft_in, _data_in, _nfft);
    CUFFT_CHECK(cufftExecC2C(_plan, cufft_in, cufft_in, CUFFT_FORWARD));
    complexMul1<<<_nfft/1024,1024>>>(cufft_in, _nfft, _dm, 296.5, 1, 50.0);
    CUFFT_CHECK(cufftExecC2C(_plan, cufft_in, cufft_in, CUFFT_INVERSE));
    scale1<<<_nfft/1024,1024>>>(cufft_in, _data_out, _nfft);
    data_out.resize(_nfft/2);
    CUDA_CHECK(cudaMemcpy(data_out.data(), &_data_out[_nfft/4], _nfft/2, cudaMemcpyDeviceToHost));
    return 0;
}

} // namespace dispersed_pulsar
} // namespace modules
} // namespace ponder
} // namespace ort
