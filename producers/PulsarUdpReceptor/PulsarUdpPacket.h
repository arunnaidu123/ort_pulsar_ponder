#ifndef ORT_PANOPTES_IO_PRODUCERS_RCPT_GSB_PulsarUdpPacket_H
#define ORT_PANOPTES_IO_PRODUCERS_RCPT_GSB_PulsarUdpPacket_H

#include "PulsarUdpHeader.h"
#include <cstdlib>
#include <cstdint>
#include <array>

/**
 * @brief
 *   Interface to packing/unpacking packets from the BeamFormer receptor stream UDP packet
 * @tparam PacketDataType datatype of the time-frequency elements
 * @tparam TimeSamplesPerPacket Number of contiguous time samples in the packet
 * @details Caution! The template parameters must match the corresponding header values
 *          This is NOT checked internally in this class.
 */

template<typename PacketDataType, unsigned TimeSamplesPerPacket>
class PulsarUdpPacket
{

    public:
        using PacketHeader = PulsarUdpHeader;
        typedef decltype(std::declval<PacketHeader>().packet_sequence_number()) PacketNumberType;


    public:
        PulsarUdpPacket();

        /**
         * @brief the total size of the udp packets header
         */
        constexpr static std::size_t header_size();

        /**
         * @brief the total size in bytes of the channel rcpt
         */
        constexpr static std::size_t payload_size();

        /**
         * @brief the total size in bytes of the channel rcpt
         */
        constexpr static std::size_t data_size();

        /**
         * @brief the total number of time samples in the packet
         */
        static std::size_t  number_of_samples();

        /**
         * @brief const Pointers to the begin and end of the data in the packet
         */
        const PacketDataType* begin() const;
        const PacketDataType* end() const;

        /**
         * @brief Pointers to the begin and end of the data in the packet
         */
        PacketDataType* begin();
        PacketDataType* end();

        PacketHeader& header();

    private: // static variables
        static constexpr std::size_t _packet_data_size = TimeSamplesPerPacket;
        static const std::size_t _number_of_samples = _packet_data_size;
        static const std::size_t _packet_payload_size = _packet_data_size;

    private:
        PacketHeader _header;
        PacketDataType _data[_packet_payload_size];
};

#include "detail/PulsarUdpPacket.cpp"

#endif // ORT_PANOPTES_IO_PRODUCERS_RCPT_GSB_PulsarUdpPacket_H
