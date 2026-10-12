#include "PulsarUdpHeader.h"

uint64_t PulsarUdpHeader::sequence_number() const
{
    return _packet_sequence_number;
}

void PulsarUdpHeader::sequence_number(uint64_t const& value)
{
    _packet_sequence_number = value;
}


uint32_t PulsarUdpHeader::packet_number() const
{
    return _packet_number;
}

void PulsarUdpHeader::packet_number(uint32_t const& value)
{
    _packet_number = value;
}

uint32_t PulsarUdpHeader::scale() const
{
    return _scale;
}

void PulsarUdpHeader::scale(uint32_t const& value)
{
    _scale = value;
}

uint32_t PulsarUdpHeader::mjd_day() const
{
    return _mjd_day;
}

void PulsarUdpHeader::mjd_day(uint32_t const& value)
{
    _mjd_day = value;
}

uint32_t PulsarUdpHeader::mjd_seconds() const
{
    return _mjd_seconds;
}

void PulsarUdpHeader::mjd_seconds(uint32_t const& value)
{
    _mjd_seconds = value;
}

uint32_t PulsarUdpHeader::mjd_nanoseconds() const
{
    return _mjd_nanoseconds;
}

void PulsarUdpHeader::mjd_nanoseconds(uint32_t const& value)
{
    _mjd_nanoseconds = value;
}