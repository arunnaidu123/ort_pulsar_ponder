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
__global__ void calPhaseKernel(double *phase, float dm, int nfft, int nbands, int block)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double dm_dispersion = 2.41e-4;
  double dispersion_per_MHz = 1e6 * dm / dm_dispersion;
  double s=6.2831853071796*dispersion_per_MHz;
  double channel_width = 16.0/((double) nbands); //bandwidth of each channel
  double bin_width = channel_width/(double)nfft;
  int t_bin = tid%nfft;
  int bin=0;
  if(t_bin<nfft/2) bin = t_bin+nfft/2;
  if(t_bin>=nfft/2) bin = t_bin-nfft/2;
  double f_mid = 334.5-((int)(tid/(nfft)))*channel_width;
  double offset = bin_width*bin - 0.5*channel_width;
  double frequency = f_mid-offset;
  double r = (offset*offset*s)/(frequency*f_mid*f_mid);

  phase[tid] = r;
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

  float2 at;
  float2 f;
  double cx,cy;
  double r = phase[tid];
  cx = cos(r);
  cy = sin(r);

  f = spectra[tid];

  at.x = (float)((f.x*cx)-(f.y*cy));
  at.y = (float)((f.x*cy)+(f.y*cx));

  spectra[tid] = at;

}

/*

__global__ void typeCaste_first(unsigned char *data0, unsigned char *data1, float2 *first, int nfft, int block)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned char pol0,pol1;
  unsigned char real,imag;


  if(tid%nfft<nfft/2)
  {
    pol0 = data0[2*tid+nfft];
    pol1 = data0[2*tid+1+nfft];

    real = ((pol0 & 0xf0)>>4);
    imag = (pol0 & 0x0f);
    first[tid].x = ((float) (int) real)-8;
    first[tid].y = ((float) (int) imag)-8;
    real = ((pol1 & 0xf0)>>4);
    imag = (pol1 & 0x0f);
    first[block+tid].x = ((float) (int) real)-8;
    first[block+tid].y = ((float) (int) imag)-8;
  }
  else
  {
    pol0 = data1[2*tid-(nfft)];
    pol1 = data1[2*tid+1-(nfft)];

    real = ((pol0 & 0xf0)>>4);
    imag = (pol0 & 0x0f);
    first[tid].x = ((float) (int) real)-8;
    first[tid].y = ((float) (int) imag)-8;
    real = ((pol1 & 0xf0)>>4);
    imag = (pol1 & 0x0f);
    first[block+tid].x = ((float) (int) real)-8;
    first[block+tid].y = ((float) (int) imag)-8;
  }

}

__global__ void typeCaste_second(unsigned char *data1, float2 *second, int nfft, int block)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned char pol0,pol1;
  unsigned char real,imag;

  pol0 = data1[2*tid];
  pol1 = data1[2*tid+1];


  real = ((pol0 & 0xf0)>>4);
  imag = (pol0 & 0x0f);

  second[tid].x = ((float) (int) real)-8;
  second[tid].y = ((float) (int)imag)-8;


  real = ((pol1 & 0xf0)>>4);
  imag = (pol1 & 0x0f);
  second[tid+block].x = ((float) (int) real)-8;
  second[tid+block].y = ((float) (int)imag)-8;
}



__global__ void typeCaste_power(float2 *first, float2 *second, unsigned char *dataOut, int nchans, int nfft, int block, float scale_factor)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  float pp,qq;
  float p0_real, p0_imag, p1_real, p1_imag;


  pp =0.0;
  qq = 0.0;


  int sample = (tid)%nfft;
  //int channel = (tid)/nfft;
  int location=0;


  if(sample<nfft/2)
  {
    location = tid+nfft/4;
    p0_real = first[location].x/(128*1024);
    p0_imag = first[location].y/(128*1024);
    p1_real = first[location+block].x/(128*1024);
    p1_imag = first[location+block].y/(128*1024);

  }
  else
  {
    location = tid-nfft/4;
    p0_real = second[location].x/(128*1024);
    p0_imag = second[location].y/(128*1024);
    p1_real = second[location+block].x/(128*1024);
    p1_imag = second[location+block].y/(128*1024);
  }

  pp = p0_real*p0_real + p0_imag*p0_imag;
  qq = p1_real*p1_real + p1_imag*p1_imag;

  pp = (pp+qq)/2.0;
  pp /= scale_factor;
  if(pp<256.0)
  dataOut[tid] = (unsigned char) (unsigned int)(pp);
  else
  {
    dataOut[tid] = 255;
  }


}

int copy_dataIn(unsigned char *data, int index)
{
  cudaMemcpy(gpu.data[(index+1)%2],data,2*host.block,cudaMemcpyHostToDevice);

  typeCaste_first<<<host.block/1024,1024>>>(gpu.data[(index)%2],gpu.data[(index+1)%2],gpu.first,gpu.nfft,host.block);
  typeCaste_second<<<host.block/1024,1024>>>(gpu.data[(index+1)%2],gpu.second,gpu.nfft,host.block);

  return 0;
}

int make_copy()
{

  cudaMemcpy(gpu.t_first,gpu.first,2*host.block*sizeof(float2),cudaMemcpyDeviceToDevice);
  cudaMemcpy(gpu.t_second,gpu.second,2*host.block*sizeof(float2),cudaMemcpyDeviceToDevice);
  return 0;
}

int dedisperse_data()
{


  checkCudaErrors(cufftExecC2C(gpu.plan, gpu.first, gpu.first, CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(gpu.plan, &gpu.first[host.block], &gpu.first[host.block], CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(gpu.plan, gpu.second, gpu.second, CUFFT_FORWARD));
  checkCudaErrors(cufftExecC2C(gpu.plan, &gpu.second[host.block], &gpu.second[host.block], CUFFT_FORWARD));


  convolve<<<host.block/(1024),1024>>>(gpu.first, gpu.second, gpu.phase, host.block);

  checkCudaErrors(cufftExecC2C(gpu.plan, gpu.first, gpu.first, CUFFT_INVERSE));
  checkCudaErrors(cufftExecC2C(gpu.plan, &gpu.first[host.block], &gpu.first[host.block], CUFFT_INVERSE));
  checkCudaErrors(cufftExecC2C(gpu.plan, gpu.second, gpu.second, CUFFT_INVERSE));
  checkCudaErrors(cufftExecC2C(gpu.plan, &gpu.second[host.block], &gpu.second[host.block], CUFFT_INVERSE));

  return 0;
}



int copy_dataOut(unsigned short *dataOut, int pol)
{
  typeCaste_power<<<host.block/(1024),1024>>>(gpu.first, gpu.second, gpu.dataOut, gpu.nchans, gpu.nfft, host.block, host.scale_factor);
  cudaMemcpy(dataOut,gpu.dataOut,host.block*sizeof(char),cudaMemcpyDeviceToHost);
  return 0;
}
*/

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort