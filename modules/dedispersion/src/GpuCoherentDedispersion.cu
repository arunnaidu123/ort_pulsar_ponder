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
    cudaMalloc((void**)&_gpu_data_out, fft_len*nchans*sizeof(float)/2);
    cudaMalloc((void**)&_gpu_in, fft_len*nchans*sizeof(float2));
    cudaMalloc((void**)&_gpu_mid, fft_len*nchans*sizeof(float2));
    cudaMalloc((void**)&_gpu_out, fft_len*nchans*sizeof(float2));
    cudaMalloc((void**)&_phase, fft_len*nchans*sizeof(double));



    int rank = 1;
    int n[1] = {(int)_gpu_fft_len};
    int howmany = _nchans;

    // Forward plan (interleaved → channel-major)
    cufftPlanMany(&_planf,
            rank, n,
            NULL, _nchans, 1,
            NULL, 1, _gpu_fft_len,
            CUFFT_C2C,
            howmany
    );

    // Inverse plan (channel-major → interleaved)
    cufftPlanMany(&_plani,
        rank, n,
        NULL, 1, _gpu_fft_len,
        NULL, _nchans, 1,
        CUFFT_C2C,
        howmany
    );
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


    cufftExecC2C(_planf, _gpu_in, _gpu_mid, CUFFT_FORWARD);

    convolve<<<block_size/1024, 1024>>>(_gpu_mid, _phase, _gpu_fft_len);

    cufftExecC2C(_planf, _gpu_mid, _gpu_out, CUFFT_INVERSE);

    typecast_out<<<block_size/2048, 1024>>>(_gpu_data_out, _gpu_out, _gpu_fft_len, block_size/2);

    cudaMemcpy(data_out.data(), &_gpu_data_out[0], sizeof(float)*block_size/2, cudaMemcpyDeviceToHost);
    //std::cout<<data_in[0].real()<<" "<<data_in[0].imag()<<" \n";
    std::cout<<data_out[0]<<" "<<data_out[1]<<" \n";

    return 0;

}


} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort