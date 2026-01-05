#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include <cstring>
#include "../GpuCoherentDedispersion.h"
#include "../kernels/kernels.h"

void cornerturn_gpu_launch(std::vector<std::complex<float>> const& , std::vector<std::complex<float>>&, int, int);


#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr,                                                    \
                "CUDA error at %s:%d: %s\n",                               \
                __FILE__, __LINE__, cudaGetErrorString(err));              \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

#define CUFFT_CHECK(call)                                                  \
do {                                                                       \
    cufftResult err = (call);                                              \
    if (err != CUFFT_SUCCESS) {                                            \
        fprintf(stderr,                                                    \
                "CUFFT error at %s:%d: %d\n",                              \
                __FILE__, __LINE__, err);                                  \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

namespace ort {
namespace ponder {
namespace modules {
namespace dedispersion {

GpuCoherentDedispersion::GpuCoherentDedispersion(int nchans, int fft_len, float dm)
    :   _nchans(nchans)
    ,   _gpu_fft_len(fft_len)
    ,   _dm(dm)
    ,   _cpu_temp(fft_len*nchans)
{
    CUDA_CHECK(cudaMalloc((void**)&_gpu_data_out, fft_len*nchans*sizeof(float)/2));
    CUDA_CHECK(cudaMalloc((void**)&_gpu_in, fft_len*nchans*sizeof(float2)));
    CUDA_CHECK(cudaMalloc((void**)&_gpu_mid, fft_len*nchans*sizeof(float2)));
    CUDA_CHECK(cudaMalloc((void**)&_gpu_out, fft_len*nchans*sizeof(float2)));
    CUDA_CHECK(cudaMalloc((void**)&_phase, fft_len*nchans*sizeof(double)));



    int rank = 1;
    int n[1] = {(int)_gpu_fft_len};
    int howmany = _nchans;

    // Forward plan (interleaved → channel-major)
    CUFFT_CHECK(cufftPlanMany(&_planf,
            rank, n,
            NULL, 1, _gpu_fft_len,
            NULL, 1, _gpu_fft_len,
            CUFFT_C2C,
            howmany
    ));

    // Inverse plan (channel-major → interleaved)
    CUFFT_CHECK(cufftPlanMany(&_plani,
        rank, n,
        NULL, 1, _gpu_fft_len,
        NULL, _nchans, 1,
        CUFFT_C2C,
        howmany
    ));
    calPhaseKernel<<<_gpu_fft_len*nchans/1024, 1024>>>(_phase, dm, fft_len, nchans);
    cudaDeviceSynchronize();
}

GpuCoherentDedispersion::~GpuCoherentDedispersion()
{
//    (_cpu_fwd);
//    fftwf_free(_cpu_inv);
    cufftDestroy(_planf);
    cufftDestroy(_plani);
    cudaFree(_gpu_data_out);
    cudaFree(_gpu_in);
    cudaFree(_gpu_mid);
    cudaFree(_gpu_out);
    cudaFree(_phase);

}

int GpuCoherentDedispersion::dedisperse(std::vector<std::complex<float>>& data_in, std::vector<float>& data_out)
{
    unsigned block_size = _gpu_fft_len*_nchans;

    cudaMemcpy(&_gpu_in[0], &_gpu_in[block_size/2], sizeof(float2)*block_size/2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&_gpu_in[block_size/2], &data_in[0], sizeof(float2)*block_size/2, cudaMemcpyHostToDevice);

    cornerturn_gpu<<<block_size/1024, 1024>>>(_gpu_in, _gpu_mid, _nchans, _gpu_fft_len);

    CUFFT_CHECK(cufftExecC2C(_planf, _gpu_mid, _gpu_mid, CUFFT_FORWARD));

    convolve<<<block_size/1024, 1024>>>(_gpu_mid, _phase, _gpu_fft_len);

    CUFFT_CHECK(cufftExecC2C(_planf, _gpu_mid, _gpu_mid, CUFFT_INVERSE));

    cornerturn_gpu<<<block_size/1024, 1024>>>(_gpu_mid, _gpu_out, _gpu_fft_len, _nchans);

    typecast_out<<<block_size/2048, 1024>>>(_gpu_data_out, _gpu_out, _gpu_fft_len, block_size/2);

    cudaMemcpy(data_out.data(), &_gpu_data_out[0], sizeof(float)*block_size/2, cudaMemcpyDeviceToHost);

    std::cout<<data_out[0]<<" "<<data_out[1]<<" \n";

    return 0;

}


} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort