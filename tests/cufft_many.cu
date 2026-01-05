#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>

// ------------------------------------------------------------
// Error checking macros
// ------------------------------------------------------------
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

// ------------------------------------------------------------
// GPU version: cuFFT PlanMany
// ------------------------------------------------------------

void cufft_many_gpu(
    int fft_len,
    int nchans,
    const std::vector<std::complex<float>>& h_in,
    std::vector<std::complex<float>>& h_out,
    int istride,
    int idist,
    int ostride,
    int odist
) {
    const int total = fft_len * nchans;

    cufftComplex *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  total * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(cufftComplex)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          total * sizeof(cufftComplex),
                          cudaMemcpyHostToDevice));

    cufftHandle plan;
    int rank = 1;
    int n[1] = { fft_len };

    CUFFT_CHECK(
        cufftPlanMany(
            &plan,
            rank, n,
            nullptr, istride, idist,      // istride, idist
            nullptr, ostride, odist,     // ostride, odist
            CUFFT_C2C,
            nchans
        )
    );

    CUFFT_CHECK(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    //CUFFT_CHECK(cufftExecC2C(plan, d_out, d_out, CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          total * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}
