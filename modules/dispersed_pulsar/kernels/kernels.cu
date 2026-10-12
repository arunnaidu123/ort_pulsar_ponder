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
__global__
void complexMul1(float2* a, int N, double dm, double fsky, int sideband, double bw)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N)
        return;

    float2 at = a[tid];

    /*
     * Real input sampled at fs = 2*bw.
     *
     * For bw = 50 MHz:
     * fs = 100 MHz
     * df = 100 MHz / N
     */
    double df = 2.0 * bw / static_cast<double>(N);

    /*
     * Fold negative-frequency FFT bins onto 0 ... bw.
     */
    double f;

    if (tid <= N / 2)
        f = static_cast<double>(tid) * df;
    else
        f = static_cast<double>(N - tid) * df;

    double s = TWOPI * dm / DFFAC;

    /*
     * The two FFT halves must have opposite phase signs so that
     * the chirp retains Hermitian symmetry and the inverse FFT
     * produces a real-valued time series.
     */
    double r = f * f * s / ((fsky + sideband * f)* fsky * fsky);

    if (tid > N / 2) r = -r;

    /*
     * Symmetric taper near the edges of the positive-frequency band.
     */
    double taper;

    if (f > 0.5 * bw)
    {
        taper = 1.0 / sqrt( 1.0 + pow(f / (0.94 * bw), 80.0));
    }
    else
    {
        taper = 1.0 / sqrt(1.0 + pow((bw - f) / (0.94 * bw), 80.0));
    }

    float2 c;

    c.x = static_cast<float>(cos(r) * taper);
    c.y = static_cast<float>(sin(r) * taper);

    float2 t;

    t.x = at.x * c.x - at.y * c.y;
    t.y = at.x * c.y + at.y * c.x;

    a[tid] = t;
}

/*
__global__ void complexMul1(float2 *a, int N,double dm, double fsky, int sideband, double bw)
{
  float2 t;
  float2 c;
  double f,s,r,taper;
  float2 at;

  s = TWOPI*dm/(DFFAC);
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  at = a[tid];

  f = tid*bw/(N/2);
  if(f > bw)
  {
    f -= bw;
    f = bw -f;
  }
  if(tid<=(N)/2) r = f*f*s/((fsky+sideband*f)*fsky*fsky);
  else r = -1*f*f*s/((fsky+sideband*f)*fsky*fsky);

  if (f > 0.5*bw) taper = 1.0/sqrt(1.0 + pow((f/(0.94*bw)),80));
  else  taper = 1.0/sqrt(1.0 + pow(((bw-f)/(0.94*bw)),80));
  c.x = (float)( cos(r) * taper );
  c.y = ( (float)( sin(r) * taper));

  if (tid < N)
  {
    t.x = ((at.x*c.x)-(at.y*c.y));
    t.y = ((at.x*c.y)+(at.y*c.x));
    a[tid].x = t.x;
    a[tid].y = t.y;
  }
}
*/
extern "C"
__global__ void typecaste1(float2 *a, char *b, int N)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N)
  {
    a[tid].x = (float)b[tid];
    a[tid].y = 0.0;
  }
}

extern "C"
__global__ void scale1( float2 *a, char *b, int N)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N)
  {
    b[tid] = (char)(float)(a[tid].x/N);
  }
}

} // namespace dedispersion
} // namespace modules
} // namespace ponder
} // namespace ort
