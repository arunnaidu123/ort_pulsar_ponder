#include <cassert>
#include <limits>
#include <iostream>

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::PulsarUdpPacket()
{
    //_header.number_of_time_samples(TimeSamplesPerPacket);
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
constexpr std::size_t PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::header_size()
{
    return sizeof(PacketHeader);
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
constexpr std::size_t PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::payload_size()
{
    return _packet_data_size;
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
constexpr std::size_t PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::data_size()
{
    return _packet_data_size;
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
std::size_t PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::number_of_samples()
{
    return _number_of_samples;
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
const PacketDataType* PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::begin() const
{
    return &_data[0];
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
const PacketDataType* PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::end() const
{
    return &_data[_packet_data_size];
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
PacketDataType* PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::begin()
{
    return &_data[0];
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
PacketDataType* PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::end()
{
    return &_data[_packet_data_size];
}

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
typename PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::PacketHeader& PulsarUdpPacket<PacketDataType, TimeSamplesPerPacket>::header()
{
    return _header;
}