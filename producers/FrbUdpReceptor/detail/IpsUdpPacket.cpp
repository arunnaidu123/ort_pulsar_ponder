#include <cassert>
#include <limits>
#include <iostream>

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::IpsUdpPacket()
{
    //_header.number_of_time_samples(NumberofElementsPerPacket);
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
constexpr std::size_t IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::header_size()
{
    return sizeof(PacketHeader);
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
constexpr std::size_t IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::payload_size()
{
    return _packet_data_size;
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
constexpr std::size_t IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::data_size()
{
    return _packet_data_size;
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
std::size_t IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::number_of_elements()
{
    return _number_of_elements;
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
const PacketDataType* IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::begin() const
{
    return &_data[0];
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
const PacketDataType* IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::end() const
{
    return &_data[_packet_data_size];
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
PacketDataType* IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::begin()
{
    return &_data[0];
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
PacketDataType* IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::end()
{
    return &_data[_packet_data_size];
}

template<typename PacketDataType, unsigned NumberofElementsPerPacket>
typename IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::PacketHeader& IpsUdpPacket<PacketDataType, NumberofElementsPerPacket>::header()
{
    return _header;
}
