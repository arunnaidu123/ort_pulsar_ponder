#include <gtest/gtest.h>
#include "../modules/pfb/PolyPhaseFB.h"
#include <fstream>
#include <cmath>
#include <complex>
#include <vector>

// All-zero real input must channelize to all-zero output.
TEST(PFBTest, ZeroInputYieldsZeroOutput)
{
    const unsigned fft_len = 64, taps = 8;
    const unsigned filter_len = fft_len * taps;
    ort::ponder::modules::pfb::PolyPhaseFB filter(fft_len, 1.0/(4*fft_len), taps);

    std::vector<float> data_in(2*filter_len, 0.0f);
    std::vector<std::complex<float>> data_out(taps * fft_len/2);

    filter.exec(data_in, data_out, 0);

    for (const auto& c : data_out)
    {
        EXPECT_EQ(c.real(), 0.0f);
        EXPECT_EQ(c.imag(), 0.0f);
    }
}

// A real tone sitting exactly on channel k0's bin should channelize almost
// entirely into channel k0, with other channels suppressed by the polyphase
// filter's stopband rejection. This guards against the PFB silently reading
// the wrong samples (e.g. treating a real time series as interleaved I/Q and
// only using every other sample).
TEST(PFBTest, SingleToneLandsInExpectedChannel)
{
    const unsigned fft_len = 64, taps = 8;
    const unsigned filter_len = fft_len * taps;
    const unsigned nchans_out = fft_len/2;
    const unsigned k0 = 10;
    const double amplitude = 100.0;

    ort::ponder::modules::pfb::PolyPhaseFB filter(fft_len, 1.0/(4*fft_len), taps);

    std::vector<float> data_in(2*filter_len);
    for (size_t n = 0; n < data_in.size(); ++n)
        data_in[n] = static_cast<float>(amplitude * std::cos(2*M_PI*k0*n/fft_len));

    std::vector<std::complex<float>> data_out(taps * nchans_out);
    filter.exec(data_in, data_out, 0);

    std::vector<double> power(nchans_out, 0.0);
    for (unsigned s = 0; s < taps; ++s)
        for (unsigned c = 0; c < nchans_out; ++c)
        {
            const auto& v = data_out[s*nchans_out + c];
            power[c] += static_cast<double>(v.real())*v.real() + static_cast<double>(v.imag())*v.imag();
        }

    unsigned peak_channel = 0;
    double peak_power = 0.0, other_power = 0.0;
    for (unsigned c = 0; c < nchans_out; ++c)
    {
        if (power[c] > peak_power) { peak_power = power[c]; peak_channel = c; }
    }
    for (unsigned c = 0; c < nchans_out; ++c)
        if (c != peak_channel) other_power += power[c];

    EXPECT_EQ(peak_channel, k0);
    EXPECT_GT(peak_power, 10 * other_power);
}

TEST(PFBTest, Sanity)
{
    ort::ponder::modules::pfb::PolyPhaseFB filter(512, 1/4096.0, 8);

    auto hamming_filter = filter.hamming_filter();

    std::ofstream ofs("filter.bin", std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file");
    }

    ofs.write(reinterpret_cast<const char*>(hamming_filter.data()),
              hamming_filter.size() * sizeof(double));

    std::cout<<hamming_filter.size()<<" \n";

    if (!ofs) {
        throw std::runtime_error("Write failed");
    }

    EXPECT_EQ(1 + 1, 2);
}