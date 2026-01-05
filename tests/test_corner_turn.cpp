#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <complex>


// Forward declarations
//void cornerturn_cpu(const float*, float*, int, int);
void cornerturn_gpu_launch(std::vector<std::complex<float>> const& , std::vector<std::complex<float>>&, int, int);

void cornerturn_cpu(
    std::vector<std::complex<float>> const& in,
    std::vector<std::complex<float>>& out,
    int number_of_channels,
    int number_of_spectra)
{
    for (int c = 0; c < number_of_channels; ++c)
        for (int s = 0; s < number_of_spectra; ++s)
            out[c * number_of_spectra + s] = in[s * number_of_channels + c];
}


bool compare_arrays(std::vector<std::complex<float>> const& h_out_cpu, std::vector<std::complex<float>> const& h_out_gpu, int number_of_channels, int number_of_spectra)
{
    // Validate
    for (int s = 0; s < number_of_spectra; ++s) {
        for (int c = 0; c < number_of_channels; ++c) {
            std::complex<float> cpu = h_out_cpu[c * number_of_spectra + s];
            std::complex<float> gpu = h_out_gpu[c * number_of_spectra + s];
            if (std::fabs(cpu - gpu) > 1e-6) {
                std::cerr << "Mismatch at (s=" << s
                          << ", c=" << c << "): "
                          << cpu << " vs " << gpu << "\n";
                return false;
            }
        }
    }

    return true;
}


TEST(CornerTurnTest, Sanity)
{
    const int number_of_channels = 256;
    const int number_of_spectra = 128*1024;

    // Host buffers
    std::vector<std::complex<float>> h_in(number_of_channels * number_of_spectra);
    std::vector<std::complex<float>> h_out_cpu(number_of_channels * number_of_spectra);
    std::vector<std::complex<float>> h_out_gpu(number_of_channels * number_of_spectra);

    // Fill input with deterministic data
    for (int c = 0; c < number_of_channels; ++c)
        for (int s = 0; s < number_of_spectra; ++s)
            h_in[s * number_of_channels + c] = {(float)(rand()%100), (float)(rand()%100)};

     // CPU reference
    cornerturn_cpu(h_in, h_out_cpu, number_of_channels, number_of_spectra);
    cornerturn_gpu_launch(h_in, h_out_gpu, number_of_channels, number_of_spectra);

    ASSERT_TRUE(compare_arrays(h_out_cpu, h_out_gpu, number_of_channels, number_of_spectra));
}