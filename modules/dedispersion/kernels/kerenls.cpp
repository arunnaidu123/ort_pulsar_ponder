#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <mutex>
#define streams 4

extern int dedisperse_data();

template<class T>
char* as_bytes(T& i)
{
  void* addr = &i;
  return static_cast<char *>(addr);
}


class gpuVariables
{
  public:
  unsigned char **data;
  float2 *dataIn;
  float2 *first;
  float2 *second;
  float2 *t_first;
  float2 *t_second;
  double *phase;
  unsigned char *dataOut;
  cufftHandle plan;
  cudaStream_t dataIn_stream[streams];
  cudaStream_t dataOut_stream[streams];
  int nchans=1024;
  int nfft = 128*1024;
  int pol=2;

  gpuVariables(long block, int nbands, int fftlen)
  {
    int batch = nbands;
    int rank = 1;
    int nRows = fftlen;
    int n[1] = {nRows};
    int idist = fftlen;
    int odist = fftlen;
    int istride = 1;
    int ostride = 1;

    for(int i=0;i<streams;i++)
    {
      cudaStreamCreate(&dataIn_stream[i]);
      cudaStreamCreate(&dataOut_stream[i]);
    }

    checkCudaErrors(cufftPlanMany(&plan, rank, n, NULL, istride, idist, NULL, ostride, odist, CUFFT_C2C, batch));
    data = (unsigned char**) malloc(2*sizeof(unsigned char*));
    checkCudaErrors(cudaMalloc((void **)&dataOut, 4*sizeof(char)*block));
    for(int i=0;i<2;i++) checkCudaErrors(cudaMalloc((void **)&data[i], 2*pol*sizeof(unsigned char)*block));
    checkCudaErrors(cudaMalloc((void **)&first, pol*sizeof(float2)*block));
    checkCudaErrors(cudaMalloc((void **)&second, pol*sizeof(float2)*block));
    checkCudaErrors(cudaMalloc((void **)&phase, sizeof(double)*block));

    nfft = fftlen;
    nchans = nbands;
  }

  ~gpuVariables()
  {
    cufftDestroy(plan);
    cudaFree(dataIn);
    cudaFree(dataOut);
    for(int i=0;i<2;i++) cudaFree(data[i]);
    cudaFree(data);
    cudaFree(phase);
    cudaFree(first);
    cudaFree(second);
  }
};

extern class filterbank fil;
extern class gpuVariables gpu;
extern class hostVariables host;