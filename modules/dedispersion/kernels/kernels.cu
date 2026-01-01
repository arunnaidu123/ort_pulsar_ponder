//#include "filterbank_cuda.hpp"
//#include "filterbank_chime.hpp"

#include <cstdio>
namespace ort {
namespace ponder {
namespace modules {
namespace dedispersion {

#define TWOPI 6.2831853071796
#define DFFAC 2.41e-10
#define DVAL 4.148808e9

extern "C"
__global__ void calPhaseKernel(double *phase, float dm, int nfft, int nbands)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int band = tid/nfft;
    int bin = tid%nfft;

    double dm_dispersion = 2.41e-4;
    double dispersion_per_MHz = 1e6 * dm / dm_dispersion;
    double s=6.2831853071796*dispersion_per_MHz;
    double channel_width = 16.0/((double) nbands); //bandwidth of each channel
    double bin_width = channel_width/(double)nfft;

    double f_mid = 334.5-band*channel_width-channel_width/2.0; // centre frequency of each band
    double offset = 0.0;
    if(bin<=nfft/2) offset = -1*(bin)*bin_width;
    else offset = (nfft-bin)*bin_width;
    double frequency = f_mid+offset;
    phase[tid] = (offset*offset*s)/(frequency*f_mid*f_mid);
}

extern "C"
__global__
void typecast_in(float2* cufft_in, char* data_in, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //if(tid==0) printf("%f and %f \n",cufft_in[size+tid].x, cufft_in[size+tid].y);
    if(tid<size)
    {
        cufft_in[size+tid].x = data_in[2*tid];
        cufft_in[size+tid].y = data_in[2*tid+1];
    }
}

extern "C"
__global__
void typecast_out(float* data_out, float2* cufft_out, int gpu_fft_len, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<size)
    {
        float2 r = cufft_out[tid+size/2];
        r.x = r.x/(gpu_fft_len);
        r.y = r.y/(gpu_fft_len);
        data_out[tid] = ((r.x)*(r.x) + (r.y)*(r.y));
    }
}

extern "C"
__global__ void convolve(float2 *spectra, double *phase, int nfft)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float2 at;
    float2 f;
    double cx,cy;
    double r = phase[tid];

    if(tid<10) printf("%1.18lf\n",r);
    cx = cos(r);
    cy = sin(r);
    f = spectra[tid];
    f.x = f.x/4096;
    f.y = f.y/4096;
    at.x = (float)((f.x*cx)-(f.y*cy));
    at.y = (float)((f.x*cy)+(f.y*cx));
    spectra[tid] = at;
}

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort