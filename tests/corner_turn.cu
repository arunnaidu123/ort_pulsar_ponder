#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_CHANNELS 256

__global__ void cornerturn_gpu(
    const float2* __restrict__ in,
    float2* __restrict__ out,
    int number_of_channels,
    int number_of_spectra)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float2 tile[MAX_CHANNELS * 16];

    for(int i=0; i<number_of_channels; ++i)
    {
        tile[number_of_channels*threadIdx.x + i] = in[blockIdx.x*blockDim.x*number_of_channels+threadIdx.x*number_of_channels+i];
    }
    //if(tid==16) printf("%d %d\n", number_of_channels*threadIdx.x, blockIdx.x*blockDim.x*number_of_channels);

    __syncthreads();

    for(int i=0; i<number_of_channels; ++i)
    {
        out[blockDim.x*gridDim.x*i+tid] = tile[number_of_channels*threadIdx.x + i];
    }
}

void cornerturn_gpu_launch(
    std::vector<std::complex<float>> const& h_in,
    std::vector<std::complex<float>>& h_out_gpu,
    int number_of_channels,
    int number_of_spectra)
{
    std::cout<<"launching kernel \n";
    const size_t in_size  = number_of_channels * number_of_spectra * sizeof(std::complex<float>);
    const size_t out_size = number_of_channels * number_of_spectra * sizeof(std::complex<float>);

    // Device buffers
    float2 *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, in_size);
    cudaMalloc(&d_out, out_size);

    std::cout<<h_in[0]<<"\n";

    cudaMemcpy(d_in, h_in.data(), in_size, cudaMemcpyHostToDevice);

    cornerturn_gpu<<<number_of_spectra/16, 16>>>(d_in, d_out, number_of_channels, number_of_spectra);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, out_size, cudaMemcpyDeviceToHost);

    std::cout<<h_out_gpu[0]<<"\n";

    cudaFree(d_in);
    cudaFree(d_out);
}