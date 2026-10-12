#ifndef ORT_OWFA_ARGUS_DATA_VOLTAGESERIES_VOLTAGEHEADER_H
#define ORT_OWFA_ARGUS_DATA_VOLTAGESERIES_VOLTAGEHEADER_H

#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <cstdint>

class VoltageHeader
{
public:
    VoltageHeader(unsigned number_of_modules, unsigned number_of_samples);
    ~VoltageHeader();

    uint64_t packet_sequence_number() const;
    void packet_sequence_number(uint64_t value);

    uint32_t packet_number() const;
    void packet_number(uint32_t value);

    uint32_t mjd_day() const;
    void  mjd_day(uint32_t value);

    uint32_t mjd_seconds() const;
    void mjd_seconds(uint32_t value);

    uint64_t mjd_nanoseconds() const;
    void mjd_nanoseconds(uint64_t value);

    uint32_t number_of_modules() const;
    void number_of_modules(uint32_t value);

    uint32_t number_of_samples() const;
    void number_of_samples(uint32_t value);

    float sampling_time() const;
    void sampling_time(float value);

private:
    uint64_t _packet_sequence_number;
    uint32_t _packet_number;
    uint32_t _mjd_day;
    uint32_t _mjd_seconds;
    uint64_t _mjd_nanoseconds;
    uint32_t _number_of_modules;
    uint32_t _number_of_samples;
    float _sampling_time;
};

#endif //ORT_OWFA_ARGUS_DATA_VOLTAGESERIES_VOLTAGEHEADER_H
