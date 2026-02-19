#pragma once

extern "C"
__global__ void complexMul1( float2 *a, int N,double dm,double fsky, int sideband,double bw);

extern "C"
__global__ void typecaste1( float2 *a, char *b, int N);

extern "C"
__global__ void scale1( float2 *a, char *b, int N);