#ifndef ORT_PULSAR_PONDER_PIPELINE_SSE_POWER_H
#define ORT_PULSAR_PONDER_PIPELINE_SSE_POWER_H

#include <complex>
#include <immintrin.h>

// Accumulate four complex bins at once while reversing channel order.
static inline void accumulate_power_sse4(const std::complex<float>* fft,
                                         float* power,
                                         unsigned nchans)
{
    const float* fft_values = reinterpret_cast<const float*>(fft);
    unsigned channel = 0;
    for(; channel + 4 <= nchans; channel += 4)
    {
        const __m128 complex01 = _mm_loadu_ps(fft_values + 2*channel);
        const __m128 complex23 = _mm_loadu_ps(fft_values + 2*channel + 4);
        const __m128 squares01 = _mm_mul_ps(complex01, complex01);
        const __m128 squares23 = _mm_mul_ps(complex23, complex23);
        __m128 magnitudes = _mm_hadd_ps(squares01, squares23);
        magnitudes = _mm_shuffle_ps(magnitudes, magnitudes, _MM_SHUFFLE(0, 1, 2, 3));

        float* destination = power + nchans - channel - 4;
        _mm_storeu_ps(destination,
                      _mm_add_ps(_mm_loadu_ps(destination), magnitudes));
    }

    for(; channel < nchans; ++channel)
    {
        const float real = fft_values[2*channel];
        const float imag = fft_values[2*channel+1];
        power[nchans-1-channel] += real*real + imag*imag;
    }
}

static inline void normalize_power_sse4(float* power, unsigned nchans, float scale)
{
    const __m128 scale4 = _mm_set1_ps(scale);
    unsigned channel = 0;
    for(; channel + 4 <= nchans; channel += 4)
    {
        _mm_storeu_ps(power + channel,
                      _mm_mul_ps(_mm_loadu_ps(power + channel), scale4));
    }
    for(; channel < nchans; ++channel)
    {
        power[channel] *= scale;
    }
}

#endif
