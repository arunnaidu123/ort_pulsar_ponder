#include <chrono>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../modules/gpu_pulsar_processing/GpuPulsarProcessing.h"
#include "../utils/FilterbankHeader.h"

using ort::ponder::modules::GpuPulsarProcessing;

namespace
{

constexpr std::size_t default_input_bytes = 32u * 1024u * 1024u;

bool parse_unsigned_value(const std::string& value, unsigned& result)
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

bool parse_device_value(const std::string& value, int& result)
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

bool parse_size(const std::string& value, std::size_t& result)
{
    std::size_t parsed = 0;
    unsigned long long number = 0;
    try
    {
        number = std::stoull(value, &parsed);
    }
    catch(const std::exception&)
    {
        return false;
    }
    if(parsed != value.size() || number == 0 ||
       number > std::numeric_limits<std::size_t>::max())
        return false;
    result = static_cast<std::size_t>(number);
    return true;
}

bool parse_float_value(const std::string& value, float& result)
{
    std::size_t parsed = 0;
    float number = 0;
    try
    {
        number = std::stof(value, &parsed);
    }
    catch(const std::exception&)
    {
        return false;
    }
    if(parsed != value.size())
        return false;
    result = number;
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <output_file> [-nchans <channels>]"
                     " [-nacc <spectra>] [-dm <pc/cm^3>] [-device <index>]"
                     " [-block-bytes <bytes>]\n";
        return 1;
    }

    unsigned nchans = 2048;
    unsigned nacc = 32;
    float dm = 26.71l;
    int device = 0;
    std::size_t block_bytes = default_input_bytes;
    for(int i = 3; i < argc; ++i)
    {
        const std::string option = argv[i];
        if(i + 1 >= argc)
        {
            std::cerr << "Incomplete option: " << option << '\n';
            return 1;
        }

        const std::string value = argv[++i];
        const bool valid =
            (option == "-nchans" && parse_unsigned_value(value, nchans)) ||
            (option == "-nacc" && parse_unsigned_value(value, nacc)) ||
            (option == "-dm" && parse_float_value(value, dm)) ||
            (option == "-device" && parse_device_value(value, device)) ||
            (option == "-block-bytes" && parse_size(value, block_bytes));
        if(!valid)
        {
            std::cerr << "Unknown or invalid option: " << option << '\n';
            return 1;
        }
    }

    if(block_bytes % 2 != 0)
    {
        std::cerr << "-block-bytes must be even because each complex int8 sample"
                     " occupies two bytes\n";
        return 1;
    }

    std::ifstream input_file(argv[1], std::ios::binary);
    std::ofstream output_file(argv[2], std::ios::binary);
    if(!input_file)
    {
        std::cerr << "Cannot open input file: " << argv[1] << '\n';
        return 1;
    }
    if(!output_file)
    {
        std::cerr << "Cannot open output file: " << argv[2] << '\n';
        return 1;
    }

    try
    {
        GpuPulsarProcessing processor(nchans, nacc, dm, device);
        ort::ponder::utils::FilterbankHeader<unsigned short> header(nchans);
        //header.tstart(61236.2123456); //B0329
        header.tstart(61241.723414);
        header.tsamp(nacc * 2.0 * nchans * 1e-8);
        header.write_header(output_file);

        std::vector<signed char> input(block_bytes);
        std::vector<unsigned short> output;
        std::size_t block_number = 0;
        while(input_file.read(reinterpret_cast<char*>(input.data()), input.size()) ||
              input_file.gcount() != 0)
        {
            const std::size_t bytes_read =
                static_cast<std::size_t>(input_file.gcount());
            if(bytes_read % 2 != 0)
                throw std::runtime_error("input ends with an incomplete complex sample");

            const auto start = std::chrono::high_resolution_clock::now();
            processor.process_raw(input.data(), bytes_read, output);
            output_file.write(reinterpret_cast<const char*>(output.data()),
                              output.size() * sizeof(unsigned short));
            if(!output_file)
                throw std::runtime_error("error writing filterbank output");

            const std::chrono::duration<double> elapsed =
                std::chrono::high_resolution_clock::now() - start;
            std::cout << "GPU " << processor.device() << " processed block "
                      << block_number++ << " (" << bytes_read << " bytes) in "
                      << elapsed.count() << " s\n";
        }
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
