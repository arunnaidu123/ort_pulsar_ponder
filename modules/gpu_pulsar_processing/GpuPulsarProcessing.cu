#include "GpuPulsarProcessing.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define WRAP_SIZE 32

namespace {


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
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

void check_cufft(cufftResult result, const char* operation)
{
    if(result != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cufft_error_string(result));
}

//__global__ void unpack_hermitian(const signed char* input, cufftComplex* output,
//                                 std::size_t complex_samples)
//{
//    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//    if(index >= complex_samples) return;
//    const std::size_t spectrum = index / 512;
//    const unsigned channel = static_cast<unsigned>(index % 512);
//    const float real = static_cast<float>(input[2 * index]);
//    const float imag = static_cast<float>(input[2 * index + 1]);
//    const std::size_t base = spectrum * 1024;
//    output[base + 512 + channel] = make_cuFloatComplex(real, imag);
//    output[base + 511 - channel] = make_cuFloatComplex(real, -imag);
//}
//
//__global__ void extract_real(const cufftComplex* input, float* output,
//                             std::size_t sample_count)
//{
//    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//    if(index < sample_count) output[index] = input[index].x;
//}

__global__ void apply_pfb(const float* input, const float* history,
                          const float* coefficients, float* output,
                          std::size_t sample_count, unsigned fft_length)
{
    constexpr unsigned taps = 5;
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= sample_count) return;
    const std::size_t time = index / fft_length;
    const unsigned phase = static_cast<unsigned>(index % fft_length);
    float filtered = 0.0f;
    for(unsigned tap = 0; tap < taps; ++tap)
    {
        const long long row = static_cast<long long>(time) - (taps - 1) + tap;
        const float sample = row < 0
            ? history[static_cast<std::size_t>(row + taps - 1) * fft_length + phase]
            : input[static_cast<std::size_t>(row) * fft_length + phase];
        filtered += coefficients[tap * fft_length + phase] * sample;
    }
    output[index] = filtered;
}

__global__ void accumulate_detected_power(const cufftComplex* spectra, unsigned short* output,
                                          std::size_t output_values,
                                          unsigned nchans, unsigned nacc)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index >= output_values) return;
    const std::size_t group = index / nchans;
    const unsigned fft_bin = nchans - 1 - static_cast<unsigned>(index % nchans);
    const std::size_t fft_stride = static_cast<std::size_t>(nchans);
    float power = 0.0f;
    for(unsigned accumulation = 0; accumulation < nacc; ++accumulation)
    {
        const std::size_t spectrum = group * nacc + accumulation;
        const cufftComplex value = spectra[spectrum * fft_stride + fft_bin];
        power += value.x * value.x + value.y * value.y;
    }
    output[index] = (unsigned short)((power / static_cast<float>(nacc*sqrt(2*nchans))));
}

__global__
void compute_filter_coeffients(double* phase, double dm, int nfft, int nbands)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nfft * nbands;

    if (tid >= total)
        return;

    const int band = tid / nfft;
    const int bin  = tid % nfft;

    constexpr double DM_CONSTANT = 2.41e-4;          // s Hz^2 pc^-1 cm^3
    constexpr double TWO_PI = 6.2831853071795864769;

    const double channel_width = 50.0 / (double)nbands;   // MHz
    const double bin_width     = channel_width / (double)nfft;

    // Centre frequency of this coarse channel (MHz)
    const double f_mid = 296.5 + band * channel_width;

    // Native cuFFT ordering:
    // 0,+df,+2df,...,+((N/2)-1)df,-N/2df,...,-df
    double offset;
    if (bin < nfft / 2)
        offset = bin * bin_width;
    else
        offset = (bin - nfft) * bin_width;

    const double frequency = f_mid + offset;

    phase[tid] = -1*TWO_PI * (1.0e6 * dm / DM_CONSTANT) * (offset * offset) / (frequency * f_mid * f_mid);
}

/*
__global__ void compute_filter_coeffients(double *phase, float dm, int nfft, int nbands)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int band = tid/nfft;
    int bin = tid%nfft;

    double dm_dispersion = 2.41e-4;
    double dispersion_per_MHz = 1e6 * dm / dm_dispersion;
    double s=6.2831853071796*dispersion_per_MHz;
    double channel_width = 50.0/((double) nbands); //bandwidth of each channel
    double bin_width = channel_width/(double)nfft;

    double f_mid = 296.5 + band*channel_width + channel_width/2.0; // centre frequency of each band
    double offset = 0.0;
    if(bin<=nfft/2) offset = (bin)*bin_width;
    else offset = -1*(nfft-bin)*bin_width;
    double frequency = f_mid+offset;
    phase[tid] = (offset*offset*s)/(frequency*f_mid*f_mid);
}
*/

__global__ void convolve(float2 *spectra, double *phase, int nfft)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float2 at;
    float2 f;
    double cx,cy;
    double r = phase[tid];

    cx = cos(r);
    cy = sin(r);
    f = spectra[tid];
    at.x = (float)((f.x*cx)-(f.y*cy))/nfft;
    at.y = (float)((f.x*cy)+(f.y*cx))/nfft;
    spectra[tid] = at;
}

__global__ void convert_int8_to_float(signed char* data_in, float* data_out)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    data_out[tid] = (float) data_in[tid];
}


__global__ void remove_extra_channel(float2 *data_in, float2 *data_out, int nchans)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int spectra = tid/nchans;
    int channel = tid%nchans;
    int location1 = spectra*(nchans+1)+channel;
    int location2 = spectra*(nchans)+channel;
    float2 at;
    at = data_in[location1];
    data_out[location2] = at;
}

__global__ void cornerturn_gpu(
    const float2* __restrict__ in,
    float2* __restrict__ out,
    int number_of_channels,
    int number_of_spectra)
{
    __shared__ float2 tile[WRAP_SIZE * WRAP_SIZE];

    int number_of_channel_tiles = number_of_channels/WRAP_SIZE;

    int tile_channel = blockIdx.x % number_of_channel_tiles;
    int tile_spetra = blockIdx.x / number_of_channel_tiles;

    int c = threadIdx.x%WRAP_SIZE;
    int s = threadIdx.x/WRAP_SIZE;

    tile[WRAP_SIZE*s + c] = in[(tile_spetra*WRAP_SIZE+s)*number_of_channels + (tile_channel*WRAP_SIZE+c)];
    __syncthreads();
    out[(tile_channel*WRAP_SIZE+c)*number_of_spectra+(tile_spetra*WRAP_SIZE+s)] = tile[WRAP_SIZE*s + c];
}

unsigned grid_size(std::size_t elements, unsigned block_size)
{
    const std::size_t blocks = (elements + block_size - 1) / block_size;
    if(blocks > std::numeric_limits<unsigned>::max())
        throw std::runtime_error("CUDA grid is too large");
    return static_cast<unsigned>(blocks);
}

} // namespace

namespace ort {
namespace ponder {
namespace modules {

GpuPulsarProcessing::GpuPulsarProcessing(unsigned nchans, unsigned nacc, float dm, int device)
    : _nchans(nchans), _nacc(nacc), _dm(dm), _filter_fftlen(0), _device(device), _input_samples(0),
      _input_spectra(0), _channelized_spectra(0), _output_values(0),
      _device_input(nullptr), _device_time_data(nullptr),
      _device_pfb_data(nullptr), _device_pfb_history(nullptr),
      _device_pfb_coefficients(nullptr), _device_channelized(nullptr),
      _tmp_device_channelized(nullptr), _corner_turned(nullptr), _device_filter(nullptr),
      _device_output(nullptr), _device_filter_coeffients(nullptr), _inverse_plan(0), _channelizer_plan(0), _coh_plan(0)
{
    if(nchans > static_cast<unsigned>(std::numeric_limits<int>::max() / 2))
        throw std::invalid_argument("nchans is too large for cuFFT");
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
}

GpuPulsarProcessing::~GpuPulsarProcessing() { release(); }

void GpuPulsarProcessing::process(const VoltageSeries<int8_t>& input,
                                  std::vector<unsigned short>& output)
{
    process_raw(reinterpret_cast<const signed char*>(&*input.cbegin()), input.size(), output);
}

void GpuPulsarProcessing::process_raw(const signed char* input,
                                      std::size_t time_samples,
                                      std::vector<unsigned short>& output)
{
    configure(time_samples);
    auto* channelized = static_cast<cufftComplex*>(_device_channelized);
    auto* tmp_channelized = static_cast<cufftComplex*>(_tmp_device_channelized);
    //auto* corner_turned = static_cast<cufftComplex*>(_corner_turned);
    check_cuda(cudaMemcpy(_device_input, input, _input_samples, cudaMemcpyHostToDevice), "copy input to GPU");
    constexpr unsigned threads = 256;
    convert_int8_to_float<<<grid_size(time_samples, threads), threads>>>(_device_input, _device_time_data);
    apply_pfb<<<grid_size(time_samples, threads), threads>>>(_device_time_data, _device_pfb_history, _device_pfb_coefficients, _device_pfb_data, time_samples, 2 * _nchans);
    check_cuda(cudaGetLastError(), "launch apply_pfb");
    constexpr std::size_t history_rows = 4;
    const std::size_t history_samples = history_rows * 2 * _nchans;
    check_cuda(cudaMemcpy(_device_pfb_history, _device_time_data + time_samples - history_samples
                         , history_samples * sizeof(float), cudaMemcpyDeviceToDevice), "update PFB history");
    check_cufft(cufftExecR2C(_channelizer_plan, _device_pfb_data, &channelized[_channelized_spectra * (_nchans + 1u)]), "execute PFB channelizer transforms");
    remove_extra_channel<<<grid_size((_nchans)*(_filter_fftlen), threads), threads>>>(reinterpret_cast<cufftComplex*>(_device_channelized), reinterpret_cast<cufftComplex*>(_tmp_device_channelized), _nchans);
    constexpr unsigned corner_threads = WRAP_SIZE * WRAP_SIZE;
    const unsigned corner_blocks = (_nchans / WRAP_SIZE) * (_filter_fftlen / WRAP_SIZE);
    std::cout<<"_filter_fftlen: "<<_filter_fftlen<<" _nchans: "<<_nchans<<"\n";
    cornerturn_gpu<<<corner_blocks, corner_threads>>>(reinterpret_cast<cufftComplex*>(_tmp_device_channelized), reinterpret_cast<cufftComplex*>(_corner_turned), _nchans, _filter_fftlen);
    check_cuda(cudaGetLastError(), "launch cornerturn_gpu");
    check_cufft(cufftExecC2C(_coh_plan, reinterpret_cast<cufftComplex*>(_corner_turned), reinterpret_cast<cufftComplex*>(_corner_turned), CUFFT_FORWARD), "channelize each band");
    compute_filter_coeffients<<<grid_size((_nchans)*(_filter_fftlen), threads), threads>>>(_device_filter_coeffients, _dm, _filter_fftlen, _nchans);
    convolve<<<grid_size((_nchans)*(_filter_fftlen), threads), threads>>>(reinterpret_cast<cufftComplex*>(_corner_turned), _device_filter_coeffients, _filter_fftlen);
    check_cufft(cufftExecC2C(_coh_plan, reinterpret_cast<cufftComplex*>(_corner_turned), reinterpret_cast<cufftComplex*>(_corner_turned), CUFFT_INVERSE), "inverse after applying the filter");
    cornerturn_gpu<<<corner_blocks, corner_threads>>>(reinterpret_cast<cufftComplex*>(_corner_turned), reinterpret_cast<cufftComplex*>(_tmp_device_channelized), _filter_fftlen, _nchans);
    accumulate_detected_power<<<grid_size(_output_values, threads), threads>>>(&tmp_channelized[_channelized_spectra * (_nchans)/2], _device_output, _output_values, _nchans, _nacc);
    check_cuda(cudaMemcpy(&channelized[0], &channelized[_channelized_spectra * (_nchans + 1u)], _channelized_spectra * (_nchans + 1u) * sizeof(cufftComplex), cudaMemcpyDeviceToDevice), "copy overlap to front of the buffer");
    check_cuda(cudaGetLastError(), "launch accumulate_detected_power");
    output.resize(_output_values);
    check_cuda(cudaMemcpy(output.data(), _device_output, _output_values * sizeof(unsigned short), cudaMemcpyDeviceToHost), "copy filterbank output from GPU");
}

int GpuPulsarProcessing::device() const { return _device; }

void cornerturn_gpu_launch(std::vector<std::complex<float>> const& h_in,
                           std::vector<std::complex<float>>& h_out,
                           int number_of_channels,
                           int number_of_spectra)
{
    if(number_of_channels % WRAP_SIZE != 0 || number_of_spectra % WRAP_SIZE != 0)
        throw std::runtime_error("cornerturn_gpu_launch: dimensions must be multiples of the tile size");
    const std::size_t elements = static_cast<std::size_t>(number_of_channels) * number_of_spectra;
    h_out.resize(elements);
    float2* d_in = nullptr;
    float2* d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, elements * sizeof(float2)), "allocate corner-turn input");
    check_cuda(cudaMalloc(&d_out, elements * sizeof(float2)), "allocate corner-turn output");
    check_cuda(cudaMemcpy(d_in, h_in.data(), elements * sizeof(float2), cudaMemcpyHostToDevice), "copy corner-turn input to GPU");
    constexpr unsigned corner_threads = WRAP_SIZE * WRAP_SIZE;
    const unsigned corner_blocks = static_cast<unsigned>((number_of_channels / WRAP_SIZE) * (number_of_spectra / WRAP_SIZE));
    cornerturn_gpu<<<corner_blocks, corner_threads>>>(d_in, d_out, number_of_channels, number_of_spectra);
    check_cuda(cudaGetLastError(), "launch cornerturn_gpu");
    check_cuda(cudaMemcpy(h_out.data(), d_out, elements * sizeof(float2), cudaMemcpyDeviceToHost), "copy corner-turn output from GPU");
    cudaFree(d_in);
    cudaFree(d_out);
}

void GpuPulsarProcessing::configure(std::size_t time_samples)
{
    if(time_samples == _input_samples) return;
    unsigned fft_length = 2*_nchans;
    constexpr std::size_t pfb_taps = 5;
    std::vector<float> preserved_history;
    if(_device_pfb_history)
    {
        preserved_history.resize((pfb_taps - 1) * fft_length);
        check_cuda(cudaMemcpy(preserved_history.data(), _device_pfb_history,
                              preserved_history.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "preserve PFB history while resizing GPU buffers");
    }
    release();

    if(time_samples % fft_length != 0)
        throw std::runtime_error("input buffer does not contain complete channelizer spectra");
    _channelized_spectra = time_samples / fft_length;
    if(_channelized_spectra % _nacc != 0)
        throw std::runtime_error("input buffer does not contain complete accumulation groups");
    _output_values = (_channelized_spectra / _nacc) * _nchans;
    if(_channelized_spectra < pfb_taps - 1)
        throw std::runtime_error("input buffer is too short for the five-tap PFB");
    if(_input_spectra > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       _channelized_spectra > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("cuFFT batch size exceeds INT_MAX");

    _filter_fftlen = 2*_channelized_spectra;
    if(_nchans % WRAP_SIZE != 0 || _filter_fftlen % WRAP_SIZE != 0)
        throw std::runtime_error("nchans and filter FFT length must be multiples of the corner-turn tile size");

    try
    {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_input), time_samples * sizeof(signed char)), "allocate GPU input");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_time_data), time_samples * sizeof(float)), "allocate time-domain buffer");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_pfb_data), time_samples * sizeof(float)), "allocate PFB output buffer");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_pfb_history), (pfb_taps - 1) * fft_length * sizeof(float)), "allocate PFB history");
        if(preserved_history.empty())
            check_cuda(cudaMemset(_device_pfb_history, 0, (pfb_taps - 1) * fft_length * sizeof(float)), "initialize PFB history");
        else
            check_cuda(cudaMemcpy(_device_pfb_history, preserved_history.data(), preserved_history.size() * sizeof(float), cudaMemcpyHostToDevice), "restore PFB history after resizing GPU buffers");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_pfb_coefficients), pfb_taps * fft_length * sizeof(float)), "allocate PFB coefficients");
        check_cuda(cudaMalloc(&_device_channelized, 2*_channelized_spectra * (_nchans + 1u) * sizeof(cufftComplex)), "allocate channelizer output");
        check_cuda(cudaMalloc(&_tmp_device_channelized, 2*_channelized_spectra * (_nchans) * sizeof(cufftComplex)), "allocate channelizer after removing the extra channel");
        check_cuda(cudaMalloc(&_corner_turned, 2*_channelized_spectra * (_nchans) * sizeof(cufftComplex)), "allocate channelizer after removing the extra channel");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_output), _output_values * sizeof(unsigned short)), "allocate GPU output");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&_device_filter_coeffients), (_nchans + 1u)*(_filter_fftlen) * sizeof(double)), "allocate GPU output");


        std::vector<float> coefficients(pfb_taps * fft_length);
        const double cutoff = 1.0 / (4.0 * fft_length);
        const double pi = 3.14159265358979323846;
        const double midpoint = (static_cast<double>(coefficients.size()) - 1.0) / 2.0;
        for(std::size_t tap = 0; tap < pfb_taps; ++tap)
        {
            for(std::size_t phase = 0; phase < fft_length; ++phase)
            {
                const std::size_t i = tap * fft_length + phase;
                const double x = static_cast<double>(i) - midpoint;
                const double sinc = x == 0.0 ? 1.0 : std::sin(2.0 * pi * cutoff * x) / (2.0 * pi * cutoff * x);
                const double window = 0.54 - 0.46 * std::cos(2.0 * pi * i / (static_cast<double>(coefficients.size()) - 1.0));
                coefficients[i] = static_cast<float>(sinc * window);
            }
        }

        check_cuda(cudaMemcpy(_device_pfb_coefficients, coefficients.data(), coefficients.size() * sizeof(float), cudaMemcpyHostToDevice), "copy PFB coefficients to GPU");

        int channelizer_length[] = {static_cast<int>(2 * _nchans)};
        check_cufft(cufftPlanMany(&_channelizer_plan, 1, channelizer_length, nullptr, 1, static_cast<int>(2 * _nchans),
                                  nullptr, 1, static_cast<int>(_nchans + 1), CUFFT_R2C, static_cast<int>(_channelized_spectra)), "create channelizer FFT plan");

        int filter_channelizer_length[] = {static_cast<int>(_filter_fftlen)};
        check_cufft(cufftPlanMany(&_coh_plan, 1, filter_channelizer_length, nullptr, 1, static_cast<int>(_filter_fftlen)
                                 , nullptr, 1, static_cast<int>(_filter_fftlen), CUFFT_C2C, static_cast<int>(_nchans)), "create channelizer forward FFT plan");

        _input_samples = time_samples;
    }
    catch(...)
    {
        release();
        throw;
    }
}

void GpuPulsarProcessing::release() noexcept
{
    if(_coh_plan) cufftDestroy(_coh_plan);
    if(_channelizer_plan) cufftDestroy(_channelizer_plan);
    if(_device_input) cudaFree(_device_input);
    if(_device_time_data) cudaFree(_device_time_data);
    if(_device_pfb_data) cudaFree(_device_pfb_data);
    if(_device_pfb_history) cudaFree(_device_pfb_history);
    if(_device_pfb_coefficients) cudaFree(_device_pfb_coefficients);
    if(_device_channelized) cudaFree(_device_channelized);
    if(_tmp_device_channelized) cudaFree(_tmp_device_channelized);
    if(_corner_turned) cudaFree(_corner_turned);
    if(_device_output) cudaFree(_device_output);
    if(_device_filter_coeffients) cudaFree(_device_filter_coeffients);
    _coh_plan = _channelizer_plan = 0;
    _device_input = nullptr;
    _device_time_data = _device_pfb_data = _device_pfb_history = nullptr;
    _device_pfb_coefficients = nullptr;
    _device_channelized = nullptr;
    _tmp_device_channelized = nullptr;
    _corner_turned = nullptr;
    _device_output = nullptr;
    _device_filter_coeffients = nullptr;
    _input_spectra = _channelized_spectra = _output_values = 0;
}

} // namespace modules
} // namespace ponder
} // namespace ort
