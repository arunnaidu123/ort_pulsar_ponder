#ifndef ORT_PONDER_MODULES_PFB_POLYFACEFB_H
#define ORT_PONDER_MODULES_PFB_POLYFACEFB_H

#include <iostream>
#include <fstream>
#include <vector>
#include <fftw3.h>
#include <complex>

namespace ort {
namespace ponder {
namespace modules {
namespace pfb {

class PolyPhaseFB
{

public:
    PolyPhaseFB() = delete;
    PolyPhaseFB(unsigned fft_len, double cutoff_frac, unsigned taps);

    ~PolyPhaseFB();

    void exec(std::vector<char> const& data_in, std::vector<std::complex<float>>& data_out, unsigned start_index);

    double dot(const std::vector<double>& a, const std::vector<double>& b);

    std::vector<double>& hamming_filter();

private:
    unsigned _filter_len;
    unsigned _fft_len;
    unsigned _taps;
    double _cutoff_frac;
    std::vector<double> _hamming_fir;
    std::vector<std::vector<double>> _ham_poly;
    std::vector<fftw_complex> _fft_in;
    std::vector<fftw_complex> _fft_out;
    fftw_plan _plan;
};

} // namespace pfb
} // namespace utils
} // namespace ponder
} // namespace ort

#endif //ORT_PONDER_MODULES_PFB_POLYFACEFB_H