#include <gtest/gtest.h>
#include <complex>
#include <cstdlib>
#include <vector>

namespace ort {
namespace ponder {
namespace modules {

void cornerturn_gpu_launch(std::vector<std::complex<float>> const& h_in,
                           std::vector<std::complex<float>>& h_out,
                           int number_of_channels,
                           int number_of_spectra);

} // namespace modules
} // namespace ponder
} // namespace ort

namespace {

void cornerturn_cpu(std::vector<std::complex<float>> const& in,
                    std::vector<std::complex<float>>& out,
                    int number_of_channels,
                    int number_of_spectra)
{
    for(int c = 0; c < number_of_channels; ++c)
        for(int s = 0; s < number_of_spectra; ++s)
            out[c * number_of_spectra + s] = in[s * number_of_channels + c];
}

void run_corner_turn_test(int number_of_channels, int number_of_spectra)
{
    const std::size_t elements = static_cast<std::size_t>(number_of_channels) * number_of_spectra;
    std::vector<std::complex<float>> h_in(elements);
    for(std::size_t i = 0; i < elements; ++i)
        h_in[i] = {static_cast<float>(rand() % 100), static_cast<float>(rand() % 100)};

    std::vector<std::complex<float>> h_out_cpu(elements);
    cornerturn_cpu(h_in, h_out_cpu, number_of_channels, number_of_spectra);

    std::vector<std::complex<float>> h_out_gpu;
    ort::ponder::modules::cornerturn_gpu_launch(h_in, h_out_gpu, number_of_channels, number_of_spectra);

    ASSERT_EQ(h_out_gpu.size(), elements);
    for(int c = 0; c < number_of_channels; ++c)
        for(int s = 0; s < number_of_spectra; ++s)
        {
            const std::size_t index = static_cast<std::size_t>(c) * number_of_spectra + s;
            ASSERT_EQ(h_out_cpu[index], h_out_gpu[index])
                << "mismatch at channel " << c << ", spectrum " << s;
        }
}

} // namespace

TEST(GpuPulsarCornerTurnTest, SingleTile)
{
    run_corner_turn_test(32, 32);
}

TEST(GpuPulsarCornerTurnTest, RectangularMultiTile)
{
    // Non-square tile counts (2 channel tiles x 37 spectra tiles) catch grid/block
    // configuration mismatches that a square case would hide.
    run_corner_turn_test(64, 32 * 37);
}

TEST(GpuPulsarCornerTurnTest, LargeGrid)
{
    run_corner_turn_test(512, 64 * 1024);
}
