#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include "FrequencyHeader.h"

FrequencyHeader::FrequencyHeader(unsigned number_of_modules, unsigned number_of_spectra, unsigned number_of_channels)
: _mjd_day(0)
, _mjd_seconds(0)
, _mjd_nanoseconds(0)
, _number_of_modules(number_of_modules)
, _number_of_spectra(number_of_spectra)
, _number_of_channels(number_of_channels)
{
}

FrequencyHeader::~FrequencyHeader()
{

}

uint64_t FrequencyHeader::packet_sequence_number() const
{
    return _packet_sequence_number;
}

void FrequencyHeader::packet_sequence_number(uint64_t value)
{
    _packet_sequence_number = value;
}

uint32_t FrequencyHeader::packet_number() const
{
    return _packet_number;
}

void FrequencyHeader::packet_number(uint32_t value)
{
    _packet_number = value;
}

uint32_t FrequencyHeader::mjd_day() const
{
    return _mjd_day;
}

void  FrequencyHeader::mjd_day(uint32_t value)
{
    _mjd_day = value;
}

uint32_t FrequencyHeader::mjd_seconds() const
{
    return _mjd_seconds;
}

void FrequencyHeader::mjd_seconds(uint32_t value)
{
    _mjd_seconds = value;
}

uint64_t FrequencyHeader::mjd_nanoseconds() const
{
    return _mjd_nanoseconds;
}

void FrequencyHeader::mjd_nanoseconds(uint64_t value)
{
    _mjd_nanoseconds = value;
}

uint32_t FrequencyHeader::number_of_modules() const
{
    return _number_of_modules;
}

void FrequencyHeader::number_of_modules(uint32_t value)
{
    _number_of_modules = value;
}

uint32_t FrequencyHeader::number_of_spectra() const
{
    return _number_of_spectra;
}

void FrequencyHeader::number_of_spectra(uint32_t value)
{
    _number_of_spectra = value;
}

uint32_t FrequencyHeader::number_of_channels() const
{
    return _number_of_channels;
}

void FrequencyHeader::number_of_channels(uint32_t value)
{
    _number_of_channels = value;
}

float FrequencyHeader::sampling_time() const
{
    return _sampling_time;
}

void FrequencyHeader::sampling_time(float value)
{
    _sampling_time = value;
}