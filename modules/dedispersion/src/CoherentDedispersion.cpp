#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
//#include <cuda_runtime.h>
//#include <cufft.h>
//#include <cufftw.h>
#include <cstring>
#include "../CoherentDedispersion.h"
//#include "../kernels/kernels.h"

namespace ort {
namespace ponder {
namespace modules {
namespace dedispersion {

CoherentDedispersion::CoherentDedispersion(unsigned nchans, unsigned fft_len, float dm)
    :   _nchans(nchans)
    ,   _gpu_fft_len(fft_len)
    ,   _dm(dm)
    ,   _phase(nchans*fft_len)
    ,   _cpu_in(fft_len*nchans)
    ,   _cpu_mid(fft_len*nchans)
    ,   _cpu_out(fft_len*nchans)
{
    int rank = 1;
    int n[1] = {_gpu_fft_len};
    int howmany = _nchans;

    // Forward plan (interleaved → channel-major)
    _cpu_fwd = fftwf_plan_many_dft(
        rank, n, howmany,
        _cpu_in.data(),  NULL, _nchans, 1,
        _cpu_mid.data(), NULL, 1,      _gpu_fft_len,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    // Inverse plan (channel-major → interleaved)
    _cpu_inv = fftwf_plan_many_dft(
        rank, n, howmany,
        _cpu_mid.data(), NULL, 1,      _gpu_fft_len,
        _cpu_out.data(), NULL, _nchans, 1,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );
    calPhase();
}

CoherentDedispersion::~CoherentDedispersion()
{
    fftwf_free(_cpu_fwd);
    fftwf_free(_cpu_inv);
}

int CoherentDedispersion::dedisperse(std::vector<std::complex<float>>& data_in, std::vector<float>& data_out)
{
    unsigned block_size = _gpu_fft_len*_nchans;
    std::memcpy(&_cpu_in[0], &_cpu_in[block_size/2], sizeof(fftwf_complex)*block_size/2);
    std::memcpy(&_cpu_in[block_size/2], &data_in[0], sizeof(fftwf_complex)*block_size/2);

    fftwf_execute(_cpu_fwd);

    for(unsigned int i=0; i<_cpu_mid.size(); ++i)
    {
        float atx, aty;
        atx = _cpu_mid[i][0]*_phase[i].real() - _cpu_mid[i][1]*_phase[i].imag();
        aty = _cpu_mid[i][0]*_phase[i].imag() + _cpu_mid[i][1]*_phase[i].real();
        _cpu_mid[i][0] = atx;
        _cpu_mid[i][1] = aty;
    }

    fftwf_execute(_cpu_inv);

    for(unsigned int i=0; i<data_out.size(); ++i)
    {
        float cx = _cpu_out[i][0]/_gpu_fft_len;
        float cy = _cpu_out[i][1]/_gpu_fft_len;
        data_out[i] = cx*cx + cy*cy;
    }

    return 0;
}


int CoherentDedispersion::calPhase()
{
    int nbands = _nchans;
    int nfft = _gpu_fft_len;
    double dm_dispersion = 2.41e-4;
    double dispersion_per_MHz = 1e6 * _dm / dm_dispersion;
    double s=6.2831853071796*dispersion_per_MHz;
    double channel_width = 16.0/((double) nbands); //bandwidth of each channel
    double bin_width = channel_width/(double)nfft;
    for(unsigned band=0; band<_nchans; ++band)
    {
        for(int bin=0; bin<_gpu_fft_len; ++bin)
        {
            //int t_bin = bin;
            //if(bin>nfft/2) t_bin = bin-nfft/2;
            //t_bin *= -1;
            double f_mid = 334.5-band*channel_width-channel_width/2.0; // centre frequency of each band
            double offset = 0.0;

            if(bin<=_gpu_fft_len/2) offset = -1*(bin)*bin_width;
            else offset = (_gpu_fft_len-bin)*bin_width;
            double frequency = f_mid+offset;
            double r = (offset*offset*s)/(frequency*f_mid*f_mid);
            _phase[band*nfft+bin].real(std::cos(r));
            _phase[band*nfft+bin].imag(std::sin(r));
            //if(band==0) std::cout<<frequency<<" "<<offset<<" \n";
        }
    }
    return 0;
}

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort