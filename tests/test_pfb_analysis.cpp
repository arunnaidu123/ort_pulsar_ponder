#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include <cstdlib>
#include <cmath>

void pfb_analysis_launch(
    std::vector<std::complex<float>>& h_out,
    const std::vector<float>& h_in,
    const std::vector<float>& h_delay,
    const std::vector<float>& h_coeffs,
    int nchans, int nfft, int ntaps);

// CPU reference — mirrors the kernel logic exactly.
// Input and delay are real (C2R synthesis output); output is complex with zero imaginary.
static void pfb_analysis_cpu(
    std::vector<std::complex<float>>& out,
    const std::vector<float>& in,
    const std::vector<float>& delay,
    const std::vector<float>& coeffs,
    int nchans, int nfft, int ntaps)
{
    for (int t = 0; t < nfft; ++t) {
        for (int k = 0; k < nchans; ++k) {
            float acc = 0.0f;
            for (int p = 0; p < ntaps; ++p) {
                int row = t - (ntaps - 1) + p;
                float h = coeffs[p * nchans + k];
                float x = (row < 0) ? delay[(ntaps - 1 + row) * nchans + k]
                                    : in[row * nchans + k];
                acc += h * x;
            }
            out[t * nchans + k] = {acc, 0.0f};
        }
    }
}

// GPU output matches CPU reference on random real input, delay, and coefficients.
TEST(PfbAnalysisKernel, CpuGpuMatch)
{
    const int nchans = 16, nfft = 64, ntaps = 5;
    srand(42);
    auto rf = [] { return (rand() % 200 - 100) / 10.0f; };

    std::vector<float> h_in(nfft * nchans);
    std::vector<float> h_delay((ntaps - 1) * nchans);
    std::vector<float> h_coeffs(ntaps * nchans);
    for (auto& v : h_in)     v = rf();
    for (auto& v : h_delay)  v = rf();
    for (auto& v : h_coeffs) v = rf();

    std::vector<std::complex<float>> cpu_out(nfft * nchans);
    std::vector<std::complex<float>> gpu_out(nfft * nchans);

    pfb_analysis_cpu(cpu_out, h_in, h_delay, h_coeffs, nchans, nfft, ntaps);
    pfb_analysis_launch(gpu_out, h_in, h_delay, h_coeffs, nchans, nfft, ntaps);

    for (int i = 0; i < nfft * nchans; ++i) {
        EXPECT_NEAR(cpu_out[i].real(), gpu_out[i].real(), 1e-4f) << "index " << i;
        EXPECT_EQ(gpu_out[i].imag(), 0.0f) << "index " << i;
    }
}

// All-zero coefficients must produce all-zero output regardless of input.
TEST(PfbAnalysisKernel, ZeroCoefficients)
{
    const int nchans = 8, nfft = 16, ntaps = 5;
    std::vector<float> h_in(nfft * nchans, 1.0f);
    std::vector<float> h_delay((ntaps - 1) * nchans, 1.0f);
    std::vector<float> h_coeffs(ntaps * nchans, 0.0f);
    std::vector<std::complex<float>> gpu_out(nfft * nchans);

    pfb_analysis_launch(gpu_out, h_in, h_delay, h_coeffs, nchans, nfft, ntaps);

    for (int i = 0; i < nfft * nchans; ++i) {
        EXPECT_EQ(gpu_out[i].real(), 0.0f);
        EXPECT_EQ(gpu_out[i].imag(), 0.0f);
    }
}

// The delay buffer is read for time rows that precede the current block.
// With all input zeroed, only the delay contributes.
// Tap p=0 with coefficient 1 reads delay row 0 for output row t=0:
//   row = t-(ntaps-1)+p = 0-4+0 = -4  ->  delay[(4-4)*nchans+k] = delay[0*nchans+k]
// For t>=1 with the same tap, the delay row differs (non-zero only at index 0),
// so those outputs are zero.
TEST(PfbAnalysisKernel, DelayBufferContributes)
{
    const int nchans = 4, nfft = 8, ntaps = 5;
    std::vector<float> h_in(nfft * nchans, 0.0f);
    std::vector<float> h_delay((ntaps - 1) * nchans, 0.0f);
    std::vector<float> h_coeffs(ntaps * nchans, 0.0f);

    // Delay row 0 = 2.0, all other delay rows = 0.
    for (int k = 0; k < nchans; ++k)
        h_delay[0 * nchans + k] = 2.0f;

    // Only tap p=0 is active.
    for (int k = 0; k < nchans; ++k)
        h_coeffs[0 * nchans + k] = 1.0f;

    std::vector<std::complex<float>> gpu_out(nfft * nchans);
    pfb_analysis_launch(gpu_out, h_in, h_delay, h_coeffs, nchans, nfft, ntaps);

    for (int k = 0; k < nchans; ++k) {
        EXPECT_NEAR(gpu_out[0 * nchans + k].real(), 2.0f, 1e-5f);
        EXPECT_EQ(gpu_out[0 * nchans + k].imag(), 0.0f);
    }
    for (int t = 1; t < nfft; ++t) {
        for (int k = 0; k < nchans; ++k) {
            EXPECT_EQ(gpu_out[t * nchans + k].real(), 0.0f) << "t=" << t;
            EXPECT_EQ(gpu_out[t * nchans + k].imag(), 0.0f) << "t=" << t;
        }
    }
}
