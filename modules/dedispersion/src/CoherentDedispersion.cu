#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include "../CoherentDedispersion.h"
#include "../kernels/kernels.h"

namespace ort {
namespace ponder {
namespace modules {
namespace dedispersion {

CoherentDedispersion::CoherentDedispersion(unsigned nchans, unsigned fft_len, float dm)
    :   _nchans(nchans)
    ,   _gpu_fft_len(fft_len)
    ,   _dm(dm)
{
    int batch = nchans;
    int rank = 1;
    int nRows = _gpu_fft_len;
    std::vector<int> n{nRows};
    int idist = _gpu_fft_len;
    int odist = _gpu_fft_len;
    int istride = 1;
    int ostride = 1;

    cudaMalloc((void **)&_cufft_in, sizeof(float2)*_gpu_fft_len*_nchans);
    cudaMalloc((void **)&_cufft_out, sizeof(float2)*_gpu_fft_len*_nchans);
    cudaMalloc((void **)&_data_out, sizeof(float)*_gpu_fft_len*_nchans/2); //
    cudaMalloc((void **)&_data_in, sizeof(char)*_gpu_fft_len*_nchans); //data_in has both imaginary and real parts and its actual size will be fft_len/2
    cudaMalloc((void **)&_phase, sizeof(double)*_gpu_fft_len*_nchans);
    cufftPlanMany(&_plan, rank, n.data(), NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batch);

    //calPhase(dm);
}

CoherentDedispersion::~CoherentDedispersion()
{
    cufftDestroy(_plan);
    cudaFree(_cufft_in);
    cudaFree(_cufft_out);
    cudaFree(_data_in);
    cudaFree(_data_out);
    cudaFree(_phase);
}

int CoherentDedispersion::calPhase(float dm)
{
    calPhaseKernel<<<_gpu_fft_len*_nchans/(1024),1024>>>(_phase, _dm, _gpu_fft_len, _nchans, _gpu_fft_len*_nchans);

    cudaDeviceSynchronize();

    return 0;
}

int CoherentDedispersion::dedisperse(std::vector<std::complex<float>>& data_in, std::vector<float>& data_out)
{
    float2* cufft_in = reinterpret_cast<float2*>(_cufft_in);
    float2* cufft_out = reinterpret_cast<float2*>(_cufft_out);
    unsigned block_size = _gpu_fft_len*_nchans;

    //cudaMemcpy(_data_in, data_in.data(), sizeof(char)*block_size, cudaMemcpyHostToDevice);
    cudaMemcpy(&cufft_in[0], &cufft_in[block_size/2], sizeof(float2)*block_size/2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(&cufft_in[block_size/2], &data_in[block_size/2], sizeof(float2)*block_size/2, cudaMemcpyHostToDevice);
    //typecast_in<<<block_size/(2048),1024>>>(cufft_in, _data_in, block_size/2);
    cufftExecC2C(_plan, cufft_in, cufft_out, CUFFT_FORWARD);
    cufftExecC2C(_plan, cufft_out, cufft_out, CUFFT_INVERSE);
    typecast_out<<<block_size/(2048),1024>>>(_data_out, cufft_out, _gpu_fft_len, block_size/2);
    cudaMemcpy(data_out.data(), _data_out, sizeof(char)*block_size/2, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return 0;
}

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort