#pragma once

extern "C"
__global__ void complexMul( float2 *a, int N,double dm,double fsky, int sideband,double bw);

extern "C"
__global__ void typecaste( float2 *a, char *b, int N);

extern "C"
__global__ void scale( float2 *a, char *b, int N);