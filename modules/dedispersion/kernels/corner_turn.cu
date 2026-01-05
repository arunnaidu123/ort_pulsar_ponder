#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#define WRAP_SIZE 32

__global__ void cornerturn_gpu(
    const float2* __restrict__ in,
    float2* __restrict__ out,
    int number_of_channels,
    int number_of_spectra)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float2 tile[WRAP_SIZE * WRAP_SIZE];

    int number_of_channel_tiles = number_of_channels/WRAP_SIZE;

    int tile_channel = blockIdx.x % number_of_channel_tiles;
    int tile_spetra = blockIdx.x / number_of_channel_tiles;

    int c = threadIdx.x%WRAP_SIZE;
    int s = threadIdx.x/WRAP_SIZE;

    tile[WRAP_SIZE*s + c] = in[(tile_spetra*WRAP_SIZE+s)*number_of_channels + (tile_channel*WRAP_SIZE+c)];
    __syncthreads();
    out[(tile_channel*WRAP_SIZE+c)*number_of_spectra+(tile_spetra*WRAP_SIZE+s)] = tile[WRAP_SIZE*s + c];
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

    cornerturn_gpu<<<number_of_spectra*number_of_channels/(WRAP_SIZE*WRAP_SIZE), WRAP_SIZE*WRAP_SIZE>>>(d_in, d_out, number_of_channels, number_of_spectra);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_gpu.data(), d_out, out_size, cudaMemcpyDeviceToHost);

    std::cout<<h_out_gpu[0]<<"\n";

    cudaFree(d_in);
    cudaFree(d_out);
}