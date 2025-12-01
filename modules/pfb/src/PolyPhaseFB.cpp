#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "../PolyPhaseFB.h"

namespace ort {
namespace ponder {
namespace modules {
namespace pfb {

PolyPhaseFB::PolyPhaseFB(unsigned fft_len, double cutoff_frac, unsigned taps)
:   _filter_len(fft_len*taps)
,   _fft_len(fft_len)
,   _taps(taps)
,   _hamming_fir(_filter_len)
,   _ham_poly(fft_len, std::vector<double>(taps))
,   _cutoff_frac(cutoff_frac)
,   _fft_in(fft_len)
,   _fft_out(fft_len)
{
    double fc = cutoff_frac;
    int M = _hamming_fir.size() - 1;

    for (int n = 0; n < _hamming_fir.size(); n++)
    {
        double x = n - M/2.0;
        double sinc = (x == 0) ? 1.0 : std::sin(2*M_PI*fc*x)/(2*M_PI*fc*x);
        double w = 0.54 - 0.46*std::cos(2*M_PI*n/M);  // Hamming window
        _hamming_fir[n] = sinc * w;
    }

    _plan = fftw_plan_dft_1d(fft_len, _fft_in.data(), _fft_out.data(), FFTW_FORWARD, FFTW_ESTIMATE);

    for (int k = 0; k < fft_len; k++) {
        for (int n = 0; n < taps; n++) {
            _ham_poly[k][n] = _hamming_fir[n*fft_len + k];
        }
    }

    std::cout<<"taps: "<<_taps<<" fft_len "<<_fft_len<<" filter_len "<<_filter_len<<" \n";
}


PolyPhaseFB::~PolyPhaseFB()
{
    fftw_destroy_plan(_plan);
}

double PolyPhaseFB::dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0;
    for (size_t i = 0; i < b.size(); i++)
        s += a[i] * b[i];
    return s;
}


void PolyPhaseFB::exec(std::vector<char> const& data_in, std::vector<std::complex<float>>& data_out, unsigned start_index)
{
    //std::cout<<"taps: "<<_taps<<" fft_len "<<_fft_len<<" filter_len "<<_filter_len<<" \n";
    const float scale_fft = std::sqrt(_fft_len);
    const float scale_ham = std::sqrt(_taps);

    for (int i = 0; i < _filter_len; i=i+_fft_len)
    {
        // Apply polyphase filters
        for (int k = 0; k < _fft_len; k++)
        {
            std::vector<double> sub(_taps);
            for (int n = 0; n < _taps; n++)
                sub[n] = data_in[2*(i+k + n*_fft_len)]; // take samples k, k+M, k+2M...


            double v = dot(_ham_poly[k], sub);


            _fft_in[k][0] = v;   // real
            _fft_in[k][1] = 0.0; // imag
        }
        // Run FFT
        fftw_execute(_plan);
        for(unsigned int s=0; s<_fft_len/2; s++)
        {
            data_out[start_index+i/2+s].real(_fft_out[s][0]);
            data_out[start_index+i/2+s].imag(_fft_out[s][1]);
        }
    }
}


} // namespace pfb
} // namespace utils
} // namespace ponder
} // namespace ort

