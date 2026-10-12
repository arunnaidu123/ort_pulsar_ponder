#ifndef ORT_PONDER_MODULES_DISPERSEDPULSAR_DISPERSEDPULSAR_H
#define ORT_PONDER_MODULES_DISPERSEDPULSAR_DISPERSEDPULSAR_H


#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cstdint>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#endif //__CUDACC__
//#include <helper_functions.h>

//#include <helper_cuda.h>

namespace ort {
namespace ponder {
namespace modules {
namespace dispersed_pulsar {

class DispersedPulsar
{

public:
    DispersedPulsar() = delete;
    DispersedPulsar(unsigned nfft, float dm);
    ~DispersedPulsar();

    // Dedisperses data_in and channelizes the result: a 1024-point real FFT
    // (via cufftw) is taken of the dedispersed time series, and only the
    // first 512 complex channels of each spectrum are kept, cast to int8_t.
    int disperse(std::vector<char>& data_in, std::vector<int8_t>& data_out);

private:
    static constexpr unsigned _channel_fft_len = 1024;
    static constexpr unsigned _num_channels = 512;

    unsigned _nfft;
    float _dm;
    std::size_t _num_spectra;
    char* _data_in;
    char* _data_out;
    void* _cufft_in;
    int _plan;

    std::vector<char> _real_time_series;
    float* _channelizer_time_buffer;
    void* _channelizer_freq_buffer;
    void* _channelizer_plan;
};

} // namespace dispersed_pulsar
} // namespace modules
} // namespace ponder
} // namespace ort


#endif //ORT_PONDER_MODULES_DISPERSEDPULSAR_DISPERSEDPULSAR_H
