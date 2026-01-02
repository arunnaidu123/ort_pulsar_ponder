// ------------------------------------------------------------
// FFTW vs cuFFT PlanMany comparison (single file)
// ------------------------------------------------------------
//
// Layout tested:
//   Input  : time-major interleaved  (t * nchans + c)
//   Output : channel-major FFTs      (c * fft_len + k)
//
// This matches your cufftPlanMany configuration exactly.
//
// ------------------------------------------------------------
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <cmath>
#include <fftw3.h>

void cufft_many_gpu(
    int fft_len,
    int nchans,
    const std::vector<std::complex<float>>& h_in,
    std::vector<std::complex<float>>& h_out
);
// ------------------------------------------------------------
// CPU reference: FFTW plan_many_dft
// ------------------------------------------------------------
void fftw_many_reference(
    int fft_len,
    int nchans,
    const std::vector<std::complex<float>>& in,
    std::vector<std::complex<float>>& out
) {
    int rank = 1;
    int n[1] = { fft_len };
    int howmany = nchans;

    int istride = 1;
    int idist   = nchans;

    int ostride = 1;
    int odist   = nchans;

    fftwf_plan plan = fftwf_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftwf_complex*>(
            const_cast<std::complex<float>*>(in.data())),
        nullptr,
        istride, idist,
        reinterpret_cast<fftwf_complex*>(out.data()),
        nullptr,
        ostride, odist,
        FFTW_FORWARD,
        FFTW_ESTIMATE
    );

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
}


// ------------------------------------------------------------
// Comparison
// ------------------------------------------------------------
bool compare_results(
    const std::vector<std::complex<float>>& a,
    const std::vector<std::complex<float>>& b,
    float tol = 1e-4f
) {
    for (size_t i = 0; i < a.size(); i++) {
        float dr = std::abs(a[i].real() - b[i].real());
        float di = std::abs(a[i].imag() - b[i].imag());
        if (dr > tol || di > tol) {
            printf("Mismatch at %zu:\n", i);
            printf("  FFTW : (%f, %f)\n", a[i].real(), a[i].imag());
            printf("  CUFFT: (%f, %f)\n", b[i].real(), b[i].imag());
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------
// Main test
// ------------------------------------------------------------
TEST(FftTest, Sanity)
{
    const int fft_len = 8;
    const int nchans  = 2;
    const int total   = fft_len * nchans;

    printf("Testing FFTW vs cuFFT PlanMany\n");
    printf("fft_len = %d, nchans = %d\n", fft_len, nchans);

    // Input: delta in channel 2 at time 0
    std::vector<std::complex<float>> input(total, {0.0f, 0.0f});
    input[1] = {1.0f, 1.0f};

    std::vector<std::complex<float>> out_fftw(total);
    std::vector<std::complex<float>> out_cufft(total);

    fftw_many_reference(fft_len, nchans, input, out_fftw);
    cufft_many_gpu(fft_len, nchans, input, out_cufft);

    for(unsigned int i=0; i<total; ++i)
    {
        std::cout<<out_fftw[i]<<" "<<out_cufft[i]<<" \n";
    }
}