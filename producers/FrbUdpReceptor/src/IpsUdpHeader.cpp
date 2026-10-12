#include "IpsUdpHeader.h"

uint64_t IpsUdpHeader::sequence_number() const
{
    return _packet_sequence_number;
}

void IpsUdpHeader::sequence_number(uint64_t const& value)
{
    _packet_sequence_number = value;
}


uint32_t IpsUdpHeader::beam_number() const
{
    return _beam_number;
}

void IpsUdpHeader::beam_number(uint32_t const& value)
{
    _beam_number = value;
}

uint32_t IpsUdpHeader::sub_sequence_number() const
{
    return _sub_sequence_number;
}

void IpsUdpHeader::sub_sequence_number(uint32_t const& value)
{
    _sub_sequence_number = value;
}

uint32_t IpsUdpHeader::mjd_day() const
{
    return _mjd_day;
}

void IpsUdpHeader::mjd_day(uint32_t const& value)
{
    _mjd_day = value;
}

uint32_t IpsUdpHeader::mjd_seconds() const
{
    return _mjd_seconds;
}

void IpsUdpHeader::mjd_seconds(uint32_t const& value)
{
    _mjd_seconds = value;
}

uint32_t IpsUdpHeader::mjd_nanoseconds() const
{
    return _mjd_nanoseconds;
}

void IpsUdpHeader::mjd_nanoseconds(uint32_t const& value)
{
    _mjd_nanoseconds = value;
}
