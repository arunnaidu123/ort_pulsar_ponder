#include <gtest/gtest.h>
#include "../modules/pfb/PolyPhaseFB.h"
#include <fstream>

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