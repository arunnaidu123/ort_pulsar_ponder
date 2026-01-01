#pragma once

extern "C"
__global__
void calPhaseKernel(double* phase, float dm, int fft_len, int nchans);

extern "C"
__global__
void typecast_in(float2* cufft_in, char* data_in, int size);

extern "C"
__global__
void typecast_out(float* data_out, float2* cufft_out, int gpu_fft_len, int size);

extern "C"
__global__
void convolve(float2 *spectra, double *phase, int nfft);