#include <cuda_runtime.h>
#include <vector>
#include <complex>

// Local copy of the pfb_analysis kernel kept in sync with
// modules/dedispersion/kernels/kernels.cu.
// Input and delay are real (float) — the synthesis IFFT (C2R) produces real samples.
// Output is float2 with zero imaginary part.
__global__ void pfb_analysis_kernel(
    float2* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ delay,
    const float* __restrict__ coeffs,
    int nchans, int nfft, int ntaps)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nfft * nchans) return;

    int t = tid / nchans;
    int k = tid % nchans;

    float acc = 0.0f;
    for (int p = 0; p < ntaps; ++p) {
        int row = t - (ntaps - 1) + p;
        float h = coeffs[p * nchans + k];
        float x;
        if (row < 0)
            x = delay[(ntaps - 1 + row) * nchans + k];
        else
            x = in[row * nchans + k];
        acc += h * x;
    }
    out[tid] = make_float2(acc, 0.0f);
}

void pfb_analysis_launch(
    std::vector<std::complex<float>>& h_out,
    const std::vector<float>& h_in,
    const std::vector<float>& h_delay,
    const std::vector<float>& h_coeffs,
    int nchans, int nfft, int ntaps)
{
    size_t out_bytes   = (size_t)nfft * nchans * sizeof(float2);
    size_t in_bytes    = (size_t)nfft * nchans * sizeof(float);
    size_t delay_bytes = (size_t)(ntaps - 1) * nchans * sizeof(float);
    size_t coeff_bytes = (size_t)ntaps * nchans * sizeof(float);

    float2 *d_out;
    float  *d_in, *d_delay, *d_coeffs;
    cudaMalloc(&d_out,    out_bytes);
    cudaMalloc(&d_in,     in_bytes);
    cudaMalloc(&d_delay,  delay_bytes);
    cudaMalloc(&d_coeffs, coeff_bytes);

    cudaMemcpy(d_in,     h_in.data(),     in_bytes,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_delay,  h_delay.data(),  delay_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs.data(), coeff_bytes, cudaMemcpyHostToDevice);

    int total = nfft * nchans;
    pfb_analysis_kernel<<<(total + 255) / 256, 256>>>(
        d_out, d_in, d_delay, d_coeffs, nchans, nfft, ntaps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in);
    cudaFree(d_delay);
    cudaFree(d_coeffs);
}
