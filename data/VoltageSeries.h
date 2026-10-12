#ifndef ORT_OWFA_ARGUS_DATA_VOLTAGESERIES_VOLTAGESERIES_H
#define ORT_OWFA_ARGUS_DATA_VOLTAGESERIES_VOLTAGESERIES_H

#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include "../utils/AlignedAllocator.h"
#include "VoltageHeader.h"

template <typename NumericalRep>
class VoltageSeries
{
public:
    typedef std::vector<NumericalRep> DataType;
    typedef typename DataType::iterator Iterator;
    typedef typename DataType::const_iterator ConstIterator;
public:
    VoltageSeries();
    VoltageSeries(unsigned number_of_modules, unsigned number_of_samples);
    ~VoltageSeries();

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

    size_t size() const;

    Iterator begin();
    ConstIterator cbegin() const;

    Iterator end();
    ConstIterator cend() const;

    float sampling_time() const;
    void sampling_time(float value);

private:
    VoltageHeader _header;
    std::vector<NumericalRep> _data;
};

#include "detail/VoltageSeries.cpp"

#endif //ORT_OWFA_ARGUS_DATA_VOLTAGESERIES_VOLTAGESERIES_H
