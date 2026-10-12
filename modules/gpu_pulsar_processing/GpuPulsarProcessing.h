#ifndef ORT_PONDER_MODULES_GPU_PULSAR_PROCESSING_H
#define ORT_PONDER_MODULES_GPU_PULSAR_PROCESSING_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../../data/VoltageSeries.h"

namespace ort {
namespace ponder {
namespace modules {

class GpuPulsarProcessing
{
public:
    GpuPulsarProcessing(unsigned nchans, unsigned nacc, float dm, int device);
    ~GpuPulsarProcessing();

    GpuPulsarProcessing(const GpuPulsarProcessing&) = delete;
    GpuPulsarProcessing& operator=(const GpuPulsarProcessing&) = delete;

    void process(const VoltageSeries<int8_t>& input, std::vector<unsigned short>& output);
    void process_raw(const signed char* input, std::size_t complex_samples,
                     std::vector<unsigned short>& output);

    int device() const;

private:
    void configure(std::size_t input_samples);
    void release() noexcept;

    unsigned _nchans;
    unsigned _nacc;
    float _dm;
    unsigned _filter_fftlen;
    int _device;
    std::size_t _input_samples;
    std::size_t _input_spectra;
    std::size_t _channelized_spectra;
    std::size_t _output_values;
    signed char* _device_input;
    float* _device_time_data;
    float* _device_pfb_data;
    float* _device_pfb_history;
    float* _device_pfb_coefficients;
    void* _device_channelized;
    void* _tmp_device_channelized;
    void* _corner_turned;
    void* _device_filter;
    unsigned short* _device_output;
    double* _device_filter_coeffients;
    int _inverse_plan;
    int _channelizer_plan;
    int _coh_plan;
};

} // namespace modules
} // namespace ponder
} // namespace ort

#endif
