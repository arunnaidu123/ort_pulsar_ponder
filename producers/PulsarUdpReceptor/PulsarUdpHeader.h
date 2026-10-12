#ifndef ORT_ARGUS_IO_EXPORTERS_PULSARUDPTRANSMITTER_PULSARUDPHEADER_H
#define ORT_ARGUS_IO_EXPORTERS_PULSARUDPTRANSMITTER_PULSARUDPHEADER_H

#include <cinttypes>

struct PulsarUdpHeader
{
    public:

        /**
         * @brief A 64-bit unsigned integer Packet Sequence Number (PSN)
         * @details PSN will be used to provide a robust method to detect lost,
         * out of order or dropped packets. Each packet is uniquely
         * identified by the Packet Sequence Number, Beam Number and
         * First channel frequency numbers.
         */
        uint64_t sequence_number() const;
        void sequence_number(uint64_t const& value);

        uint32_t packet_number() const;
        void packet_number(uint32_t const& value);

        uint32_t scale() const;
        void scale(uint32_t const& value);

        uint64_t packet_sequence_number() const;
        void packet_sequence_number(uint64_t const& value);

        uint32_t mjd_day() const;
        void  mjd_day(uint32_t const& value);

        uint32_t mjd_seconds() const;
        void  mjd_seconds(uint32_t const& value);

        uint32_t mjd_nanoseconds() const;
        void  mjd_nanoseconds(uint32_t const& value);

    private:
        uint64_t _packet_sequence_number;        /* packet sequence number */
        uint32_t _packet_number; /*packet number*/
        float _scale; /*scaling factor*/
        uint32_t _mjd_day;     /* MJD day */
        uint32_t _mjd_seconds;     /*seconds of the day*/
        float _sampling_time;
        uint64_t _mjd_nanoseconds; /*fraction is nanoseconds*/
};

#endif // ORT_ARGUS_IO_EXPORTERS_PULSARUDPTRANSMITTER_PULSARUDPHEADER_H