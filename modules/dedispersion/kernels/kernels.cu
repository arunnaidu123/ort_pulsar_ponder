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
  double dm_dispersion = 2.41e-4;
  double dispersion_per_MHz = 1e6 * dm / dm_dispersion;
  double s=6.2831853071796*dispersion_per_MHz;
  double channel_width = 16.0/((double) nbands); //bandwidth of each channel
  double bin_width = channel_width/(double)nfft;
  int band = tid%nbands;
  int t_bin = tid/nbands;
  int bin=0;
  if(t_bin<nfft/2) bin = t_bin-nfft/2;
  if(t_bin>=nfft/2) bin = t_bin+nfft/2;
  double f_mid = 334.5-band*channel_width - 0.5*channel_width; // centre frequency of each band
  double offset = bin_width*bin - 0.5*channel_width;
  double frequency = f_mid-offset;
  double r = (offset*offset*s)/(frequency*f_mid*f_mid);
  if(tid==0) printf("frequency: %d %f ",tid, frequency);
  if(tid==256) printf("frequency: %d %f ",tid, frequency);
  if(tid==512) printf("frequency: %d %f ",tid, frequency);
  phase[tid] = -1*r;
  if(bin==0) phase[tid] =0.0;
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
        r.x = r.x/gpu_fft_len;
        r.y = r.y/gpu_fft_len;
        data_out[tid] = ((r.x*r.x + r.y*r.y));
    }
}

extern "C"
__global__ void convolve(float2 *spectra, double *phase)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//  float2 at;
//  float2 f;
//  double cx,cy;
//  double r = phase[tid];
//  cx = cos(r);
//  cy = sin(r);
//
//  f = spectra[tid];
//
//  at.x = (float)((f.x*cx)-(f.y*cy));
//  at.y = (float)((f.x*cy)+(f.y*cx));
//
//  spectra[tid] = at;
  if(tid<128*1024*175)
  {
    spectra[tid].x = 0.0;
    spectra[tid].y = 0.0;
  }

}

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort