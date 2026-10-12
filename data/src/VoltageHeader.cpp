#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include "../VoltageHeader.h"

VoltageHeader::VoltageHeader(unsigned number_of_modules, unsigned number_of_samples)
: _mjd_day(0)
, _mjd_seconds(0)
, _mjd_nanoseconds(0)
, _number_of_modules(number_of_modules)
, _number_of_samples(number_of_samples)
, _sampling_time(1e-8)
{
}

VoltageHeader::~VoltageHeader()
{

}

uint64_t VoltageHeader::packet_sequence_number() const
{
    return _packet_sequence_number;
}

void VoltageHeader::packet_sequence_number(uint64_t value)
{
    _packet_sequence_number = value;
}

uint32_t VoltageHeader::packet_number() const
{
    return _packet_number;
}

void VoltageHeader::packet_number(uint32_t value)
{
    _packet_number = value;
}

uint32_t VoltageHeader::mjd_day() const
{
    return _mjd_day;
}

void  VoltageHeader::mjd_day(uint32_t value)
{
    _mjd_day = value;
}

uint32_t VoltageHeader::mjd_seconds() const
{
    return _mjd_seconds;
}

void VoltageHeader::mjd_seconds(uint32_t value)
{
    _mjd_seconds = value;
}

uint64_t VoltageHeader::mjd_nanoseconds() const
{
    return _mjd_nanoseconds;
}

void VoltageHeader::mjd_nanoseconds(uint64_t value)
{
    _mjd_nanoseconds = value;
}

uint32_t VoltageHeader::number_of_modules() const
{
    return _number_of_modules;
}

void VoltageHeader::number_of_modules(uint32_t value)
{
    _number_of_modules = value;
}

uint32_t VoltageHeader::number_of_samples() const
{
    return _number_of_samples;
}

void VoltageHeader::number_of_samples(uint32_t value)
{
    _number_of_samples = value;
}

float VoltageHeader::sampling_time() const
{
    return _sampling_time;
}

void VoltageHeader::sampling_time(float value)
{
    _sampling_time = value;
}
