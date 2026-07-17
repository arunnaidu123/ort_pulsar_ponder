#include <cuda_runtime.h>
#include <cufft.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../producers/PulsarUdpReceptor.h"
#include "../producers/RcptRingBuffer.h"
#include "../utils/FilterbankHeader.h"

namespace
{

const char* cufft_error_string(cufftResult error)
{
    switch(error)
    {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    default: return "unknown cuFFT error";
    }
}

void check_cuda(cudaError_t result, const char* operation)
{
    if(result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

void check_cufft(cufftResult result, const char* operation)
{
    if(result != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cufft_error_string(result));
}

__global__ void unpack_hermitian(const signed char* input,
                                 cufftComplex* output,
                                 std::size_t complex_samples)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= complex_samples) return;

    const std::size_t spectrum = index / 512;
    const unsigned channel = static_cast<unsigned>(index % 512);
    const float real = static_cast<float>(input[2 * index]);
    const float imag = static_cast<float>(input[2 * index + 1]);
    const std::size_t base = spectrum * 1024;

    output[base + 512 + channel] = make_cuFloatComplex(real, imag);
    output[base + 511 - channel] = make_cuFloatComplex(real, -imag);
}

__global__ void extract_real(const cufftComplex* input,
                             float* output,
                             std::size_t sample_count)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < sample_count) output[index] = input[index].x;
}

__global__ void apply_pfb(const float* input,
                          const float* history,
                          const float* coefficients,
                          float* output,
                          std::size_t sample_count,
                          unsigned fft_length)
{
    constexpr unsigned taps = 5;
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= sample_count) return;

    const std::size_t time = index / fft_length;
    const unsigned phase = static_cast<unsigned>(index % fft_length);
    float filtered = 0.0f;
    for(unsigned tap = 0; tap < taps; ++tap)
    {
        const long long row = static_cast<long long>(time) - (taps - 1) + tap;
        float sample = 0.0f;
        if(row < 0)
        {
            const std::size_t history_row =
                static_cast<std::size_t>(row + taps - 1);
            sample = history[history_row * fft_length + phase];
        }
        else
        {
            sample = input[static_cast<std::size_t>(row) * fft_length + phase];
        }
        filtered += coefficients[tap * fft_length + phase] * sample;
    }
    output[index] = filtered;
}

__global__ void accumulate_detected_power(const cufftComplex* spectra,
                                          float* output,
                                          std::size_t output_values,
                                          unsigned nchans,
                                          unsigned nacc)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= output_values) return;

    const std::size_t group = index / nchans;
    const unsigned output_channel = static_cast<unsigned>(index % nchans);
    const unsigned fft_bin = nchans - 1 - output_channel;
    const std::size_t fft_stride = static_cast<std::size_t>(nchans) + 1;
    float power = 0.0f;

    for(unsigned accumulation = 0; accumulation < nacc; ++accumulation)
    {
        const std::size_t spectrum = group * nacc + accumulation;
        const cufftComplex value = spectra[spectrum * fft_stride + fft_bin];
        power += value.x * value.x + value.y * value.y;
    }
    output[index] = power / static_cast<float>(nacc);
}

unsigned grid_size(std::size_t element_count, unsigned block_size)
{
    const std::size_t blocks = (element_count + block_size - 1) / block_size;
    if(blocks > std::numeric_limits<unsigned>::max())
        throw std::runtime_error("CUDA grid is too large");
    return static_cast<unsigned>(blocks);
}

class GpuPulsarProcessor
{
public:
    GpuPulsarProcessor(unsigned nchans, unsigned nacc, int device)
        : _nchans(nchans)
        , _nacc(nacc)
        , _device(device)
        , _input_samples(0)
        , _input_spectra(0)
        , _channelized_spectra(0)
        , _output_values(0)
        , _device_input(nullptr)
        , _device_stage_one(nullptr)
        , _device_time_data(nullptr)
        , _device_pfb_data(nullptr)
        , _device_pfb_history(nullptr)
        , _device_pfb_coefficients(nullptr)
        , _device_channelized(nullptr)
        , _device_output(nullptr)
        , _inverse_plan(0)
        , _channelizer_plan(0)
    {
        if(nchans > static_cast<unsigned>(std::numeric_limits<int>::max() / 2))
            throw std::invalid_argument("nchans is too large for cuFFT");
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
    }

    ~GpuPulsarProcessor()
    {
        release();
    }

    GpuPulsarProcessor(const GpuPulsarProcessor&) = delete;
    GpuPulsarProcessor& operator=(const GpuPulsarProcessor&) = delete;

    void process(const FrequencySeries<int8_t>& input,
                 std::vector<float>& output)
    {
        process_raw(reinterpret_cast<const signed char*>(&*input.cbegin()),
                    input.size(),
                    output);
    }

    void process_raw(const signed char* input,
                     std::size_t complex_samples,
                     std::vector<float>& output)
    {
        configure(complex_samples);
        check_cuda(cudaMemcpy(_device_input,
                              input,
                              2 * _input_samples * sizeof(signed char),
                              cudaMemcpyHostToDevice),
                   "copy input to GPU");

        constexpr unsigned threads = 256;
        unpack_hermitian<<<grid_size(_input_samples, threads), threads>>>(
            _device_input, _device_stage_one, _input_samples);
        check_cuda(cudaGetLastError(), "launch unpack_hermitian");

        check_cufft(cufftExecC2C(_inverse_plan,
                                 _device_stage_one,
                                 _device_stage_one,
                                 CUFFT_INVERSE),
                    "execute inverse 1024-point transforms");

        const std::size_t time_samples = _input_spectra * 1024;
        extract_real<<<grid_size(time_samples, threads), threads>>>(
            _device_stage_one, _device_time_data, time_samples);
        check_cuda(cudaGetLastError(), "launch extract_real");

        apply_pfb<<<grid_size(time_samples, threads), threads>>>(
            _device_time_data,
            _device_pfb_history,
            _device_pfb_coefficients,
            _device_pfb_data,
            time_samples,
            2 * _nchans);
        check_cuda(cudaGetLastError(), "launch apply_pfb");

        constexpr std::size_t history_rows = 4;
        const std::size_t history_samples = history_rows * 2 * _nchans;
        check_cuda(cudaMemcpy(_device_pfb_history,
                              _device_time_data + time_samples - history_samples,
                              history_samples * sizeof(float),
                              cudaMemcpyDeviceToDevice),
                   "update PFB history");

        check_cufft(cufftExecR2C(_channelizer_plan,
                                 _device_pfb_data,
                                 _device_channelized),
                    "execute PFB channelizer transforms");

        accumulate_detected_power<<<grid_size(_output_values, threads), threads>>>(
            _device_channelized, _device_output, _output_values, _nchans, _nacc);
        check_cuda(cudaGetLastError(), "launch accumulate_detected_power");

        output.resize(_output_values);
        check_cuda(cudaMemcpy(output.data(),
                              _device_output,
                              _output_values * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy filterbank output from GPU");
    }

    int device() const { return _device; }

private:
    void configure(std::size_t input_samples)
    {
        if(input_samples == _input_samples) return;

        constexpr std::size_t pfb_taps = 5;
        const std::size_t fft_length = 2 * static_cast<std::size_t>(_nchans);
        std::vector<float> preserved_history;
        if(_device_pfb_history)
        {
            preserved_history.resize((pfb_taps - 1) * fft_length);
            check_cuda(cudaMemcpy(preserved_history.data(),
                                  _device_pfb_history,
                                  preserved_history.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "preserve PFB history while resizing GPU buffers");
        }
        release();

        if(input_samples == 0 || input_samples % 512 != 0)
            throw std::runtime_error("input buffer size must be divisible by 512");

        _input_samples = input_samples;
        _input_spectra = input_samples / 512;
        const std::size_t time_samples = _input_spectra * 1024;
        if(time_samples % (2 * static_cast<std::size_t>(_nchans)) != 0)
            throw std::runtime_error("input buffer does not contain complete channelizer spectra");

        _channelized_spectra = time_samples / (2 * _nchans);
        if(_channelized_spectra % _nacc != 0)
            throw std::runtime_error("input buffer does not contain complete accumulation groups");
        _output_values = (_channelized_spectra / _nacc) * _nchans;

        if(_channelized_spectra < pfb_taps - 1)
            throw std::runtime_error("input buffer is too short for the five-tap PFB");

        if(_input_spectra > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
           _channelized_spectra > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("cuFFT batch size exceeds INT_MAX");

        try
        {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_input),
                                  2 * _input_samples * sizeof(signed char)),
                       "allocate GPU input");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_stage_one),
                                  time_samples * sizeof(cufftComplex)),
                       "allocate first FFT buffer");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_time_data),
                                  time_samples * sizeof(float)),
                       "allocate time-domain buffer");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_pfb_data),
                                  time_samples * sizeof(float)),
                       "allocate PFB output buffer");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_pfb_history),
                                  (pfb_taps - 1) * fft_length * sizeof(float)),
                       "allocate PFB history");
            if(preserved_history.empty())
            {
                check_cuda(cudaMemset(_device_pfb_history,
                                      0,
                                      (pfb_taps - 1) * fft_length * sizeof(float)),
                           "initialize PFB history");
            }
            else
            {
                check_cuda(cudaMemcpy(_device_pfb_history,
                                      preserved_history.data(),
                                      preserved_history.size() * sizeof(float),
                                      cudaMemcpyHostToDevice),
                           "restore PFB history after resizing GPU buffers");
            }
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_pfb_coefficients),
                                  pfb_taps * fft_length * sizeof(float)),
                       "allocate PFB coefficients");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_channelized),
                                  _channelized_spectra *
                                      (static_cast<std::size_t>(_nchans) + 1) *
                                      sizeof(cufftComplex)),
                       "allocate channelizer output");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_output),
                                  _output_values * sizeof(float)),
                       "allocate GPU output");

            std::vector<float> coefficients(pfb_taps * fft_length);
            const double cutoff = 1.0 / (4.0 * fft_length);
            const double pi = 3.14159265358979323846;
            const double midpoint =
                (static_cast<double>(coefficients.size()) - 1.0) / 2.0;
            for(std::size_t tap = 0; tap < pfb_taps; ++tap)
            {
                for(std::size_t phase = 0; phase < fft_length; ++phase)
                {
                    const std::size_t coefficient_index = tap * fft_length + phase;
                    const double x = static_cast<double>(coefficient_index) - midpoint;
                    const double sinc = x == 0.0
                        ? 1.0
                        : std::sin(2.0 * pi * cutoff * x) /
                              (2.0 * pi * cutoff * x);
                    const double window = 0.54 - 0.46 * std::cos(
                        2.0 * pi * coefficient_index /
                        (static_cast<double>(coefficients.size()) - 1.0));
                    coefficients[coefficient_index] =
                        static_cast<float>(sinc * window);
                }
            }
            check_cuda(cudaMemcpy(_device_pfb_coefficients,
                                  coefficients.data(),
                                  coefficients.size() * sizeof(float),
                                  cudaMemcpyHostToDevice),
                       "copy PFB coefficients to GPU");

            int inverse_length[] = {1024};
            check_cufft(cufftPlanMany(&_inverse_plan, 1, inverse_length,
                                      nullptr, 1, 1024,
                                      nullptr, 1, 1024,
                                      CUFFT_C2C, static_cast<int>(_input_spectra)),
                        "create inverse FFT plan");

            int channelizer_length[] = {static_cast<int>(2 * _nchans)};
            check_cufft(cufftPlanMany(&_channelizer_plan, 1, channelizer_length,
                                      nullptr, 1, static_cast<int>(2 * _nchans),
                                      nullptr, 1, static_cast<int>(_nchans + 1),
                                      CUFFT_R2C,
                                      static_cast<int>(_channelized_spectra)),
                        "create channelizer FFT plan");
        }
        catch(...)
        {
            release();
            throw;
        }
    }

    void release() noexcept
    {
        if(_inverse_plan) cufftDestroy(_inverse_plan);
        if(_channelizer_plan) cufftDestroy(_channelizer_plan);
        if(_device_input) cudaFree(_device_input);
        if(_device_stage_one) cudaFree(_device_stage_one);
        if(_device_time_data) cudaFree(_device_time_data);
        if(_device_pfb_data) cudaFree(_device_pfb_data);
        if(_device_pfb_history) cudaFree(_device_pfb_history);
        if(_device_pfb_coefficients) cudaFree(_device_pfb_coefficients);
        if(_device_channelized) cudaFree(_device_channelized);
        if(_device_output) cudaFree(_device_output);

        _inverse_plan = 0;
        _channelizer_plan = 0;
        _device_input = nullptr;
        _device_stage_one = nullptr;
        _device_time_data = nullptr;
        _device_pfb_data = nullptr;
        _device_pfb_history = nullptr;
        _device_pfb_coefficients = nullptr;
        _device_channelized = nullptr;
        _device_output = nullptr;
        _input_samples = 0;
        _input_spectra = 0;
        _channelized_spectra = 0;
        _output_values = 0;
    }

    unsigned _nchans;
    unsigned _nacc;
    int _device;
    std::size_t _input_samples;
    std::size_t _input_spectra;
    std::size_t _channelized_spectra;
    std::size_t _output_values;
    signed char* _device_input;
    cufftComplex* _device_stage_one;
    float* _device_time_data;
    float* _device_pfb_data;
    float* _device_pfb_history;
    float* _device_pfb_coefficients;
    cufftComplex* _device_channelized;
    float* _device_output;
    cufftHandle _inverse_plan;
    cufftHandle _channelizer_plan;
};

#ifndef ORT_PONDER_PULSAR_GPU_PIPELINE_ONLY

void process_stream(RcptRingBuffer<int8_t>& ring_buffer,
                    std::ofstream& output_file,
                    unsigned nchans,
                    unsigned nacc,
                    int device)
{
    GpuPulsarProcessor processor(nchans, nacc, device);
    ort::ponder::utils::FilterbankHeader<float> header(nchans);
    std::vector<float> output;
    bool header_written = false;

    while(!ring_buffer.abort())
    {
        std::shared_ptr<RcptRingBuffer<int8_t>::BufferType> buffer;
        try
        {
            buffer = ring_buffer.get_readable();
        }
        catch(const std::runtime_error&)
        {
            if(ring_buffer.abort()) break;
            throw;
        }

        if(!header_written)
        {
            const auto& data = *buffer->second;
            const long double tstart =
                static_cast<long double>(data.mjd_day()) +
                (static_cast<long double>(data.mjd_seconds()) +
                 static_cast<long double>(data.mjd_nanoseconds()) * 1e-9L) /
                    86400.0L;
            header.tstart(tstart);
            header.tsamp(nacc * 2.0 * nchans * 1e-8);
            header.write_header(output_file);
            header_written = true;
        }

        const auto start = std::chrono::high_resolution_clock::now();
        processor.process(*buffer->second, output);
        output_file.write(reinterpret_cast<const char*>(output.data()),
                          output.size() * sizeof(float));
        if(!output_file) throw std::runtime_error("error writing filterbank output");

        const std::chrono::duration<double> elapsed =
            std::chrono::high_resolution_clock::now() - start;
        std::cout << "GPU " << processor.device() << " processed one buffer in "
                  << elapsed.count() << " s\n";
    }
}

bool parse_unsigned(const std::string& value, unsigned& result)
{
    std::size_t parsed = 0;
    unsigned long number = 0;
    try
    {
        number = std::stoul(value, &parsed);
    }
    catch(const std::exception&)
    {
        return false;
    }
    if(parsed != value.size() || number == 0 ||
       number > std::numeric_limits<unsigned>::max())
        return false;
    result = static_cast<unsigned>(number);
    return true;
}

bool parse_device(const std::string& value, int& result)
{
    std::size_t parsed = 0;
    long number = 0;
    try
    {
        number = std::stol(value, &parsed);
    }
    catch(const std::exception&)
    {
        return false;
    }
    if(parsed != value.size() || number < 0 ||
       number > std::numeric_limits<int>::max())
        return false;
    result = static_cast<int>(number);
    return true;
}

#endif

} // namespace

#ifndef ORT_PONDER_PULSAR_GPU_PIPELINE_ONLY

int main(int argc, char* argv[])
{
    static_assert(sizeof(std::complex<int8_t>) == 2,
                  "GPU input transfer requires packed complex int8 samples");

    if(argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <output_file> [-nchans <channels>] [-nacc <spectra>]"
                     " [-device <index>]\n";
        return 1;
    }

    unsigned nchans = 2048;
    unsigned nacc = 32;
    int device = 0;
    for(int i = 2; i < argc; ++i)
    {
        const std::string option = argv[i];
        if(i + 1 >= argc)
        {
            std::cerr << "Incomplete option: " << option << '\n';
            return 1;
        }

        const std::string value = argv[++i];
        const bool valid =
            (option == "-nchans" && parse_unsigned(value, nchans)) ||
            (option == "-nacc" && parse_unsigned(value, nacc)) ||
            (option == "-device" && parse_device(value, device));
        if(!valid)
        {
            std::cerr << "Unknown or invalid option: " << option << '\n';
            return 1;
        }
    }

    RcptRingBuffer<int8_t> ring_buffer(8192, 4096, 128);
    PulsarUdpReceptor receptor(ring_buffer, std::string("192.168.22.112"), 2);
    std::ofstream output_file(argv[1], std::ios::binary);
    if(!output_file)
    {
        std::cerr << "Cannot open output file: " << argv[1] << '\n';
        return 1;
    }

    std::exception_ptr receptor_error;
    std::exception_ptr processor_error;
    std::thread receptor_thread([&] {
        try
        {
            receptor.run();
        }
        catch(...)
        {
            receptor_error = std::current_exception();
            ring_buffer.abort(true);
        }
    });
    std::thread processor_thread([&] {
        try
        {
            process_stream(ring_buffer, output_file, nchans, nacc, device);
        }
        catch(...)
        {
            processor_error = std::current_exception();
            ring_buffer.abort(true);
        }
    });

    receptor_thread.join();
    processor_thread.join();
    try
    {
        if(receptor_error) std::rethrow_exception(receptor_error);
        if(processor_error) std::rethrow_exception(processor_error);
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

#endif
