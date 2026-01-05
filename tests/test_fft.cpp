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
#include <random>

void cufft_many_gpu(
    int fft_len,
    int nchans,
    const std::vector<std::complex<float>>& h_in,
    std::vector<std::complex<float>>& h_out,
    int istride,
    int idist,
    int ostride,
    int odist
);
// ------------------------------------------------------------
// CPU reference: FFTW plan_many_dft
// ------------------------------------------------------------
void fftw_many_reference(
    int fft_len,
    int nchans,
    std::vector<std::complex<float>>& in,
    std::vector<std::complex<float>>& out,
    int istride,
    int idist,
    int ostride,
    int odist)
{
    int rank = 1;
    int n[1] = { fft_len };
    int howmany = nchans;

    fftwf_plan plan = fftwf_plan_many_dft(
        rank, n, howmany,
        reinterpret_cast<fftwf_complex*>(in.data()),
        nullptr, istride, idist,
        reinterpret_cast<fftwf_complex*>(out.data()),
        nullptr, ostride, odist,
        FFTW_FORWARD, FFTW_ESTIMATE
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
    float tol = 1e-3f
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

// ------------------------------------------------------------------------------------------------
// The 11.2 cuda toolkit seems to have bug. Cant pinpoint what the issue is
// but looks like the issue stems from the bug in CufftMany plan. since the NVIDIA no longer support for K20c
// we have to figure out a work around for this. Here is the test I belive should work for the 11.0 and 11.2 cuda
// versions. If this pass that means the covolution should basically work.
// ------------------------------------------------------------------------------------------------
TEST(FftTest, Sanity)
{
    const int fft_len = 128*1024;
    const int nchans  = 256;
    const int total   = fft_len * nchans;
    int istride = 1;
    int idist = fft_len;
    int ostride = 1;
    int odist = fft_len;

    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist(0.0, 1.0);
    std::vector<std::complex<float>> input(total, {0.0f, 0.0f});

    for (auto &z : input)
        z = { dist(rng), dist(rng) };

    std::vector<std::complex<float>> out_fftw(total);
    std::vector<std::complex<float>> out_cufft(total);

    cufft_many_gpu(fft_len, nchans, input, out_cufft, istride, idist, ostride, odist);
    fftw_many_reference(fft_len, nchans, input, out_fftw, istride, idist, ostride, odist);
    ASSERT_TRUE(compare_results(out_cufft, out_fftw));
}