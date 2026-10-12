#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <iostream>

template <typename NumericalRep>
VoltageSeries<NumericalRep>::VoltageSeries()
: _header(0, 0)
, _data(0, 0)
{

}

template <typename NumericalRep>
VoltageSeries<NumericalRep>::VoltageSeries(unsigned number_of_modules, unsigned number_of_samples)
: _header(number_of_modules, number_of_samples)
, _data(number_of_modules*number_of_samples, 0)
{
}

template <typename NumericalRep>
VoltageSeries<NumericalRep>::~VoltageSeries()
 {

 }

template <typename NumericalRep>
typename VoltageSeries<NumericalRep>::Iterator VoltageSeries<NumericalRep>::begin()
{
    return _data.begin();
}

template <typename NumericalRep>
typename VoltageSeries<NumericalRep>::ConstIterator VoltageSeries<NumericalRep>::cbegin() const
{
    return _data.begin();
}

template <typename NumericalRep>
typename VoltageSeries<NumericalRep>::Iterator VoltageSeries<NumericalRep>::end()
{
    return _data.end();
}

template <typename NumericalRep>
typename VoltageSeries<NumericalRep>::ConstIterator VoltageSeries<NumericalRep>::cend() const
{
    return _data.end();
}

template <typename NumericalRep>
uint32_t VoltageSeries<NumericalRep>::mjd_day() const
{
    return _header.mjd_day();
}

template <typename NumericalRep>
void  VoltageSeries<NumericalRep>::mjd_day(uint32_t value)
{
    _header.mjd_day(value);
}

template <typename NumericalRep>
uint32_t VoltageSeries<NumericalRep>::mjd_seconds() const
{
    return _header.mjd_seconds();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::mjd_seconds(uint32_t value)
{
    _header.mjd_seconds(value);
}

template <typename NumericalRep>
uint64_t VoltageSeries<NumericalRep>::mjd_nanoseconds() const
{
    return _header.mjd_nanoseconds();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::mjd_nanoseconds(uint64_t value)
{
    _header.mjd_nanoseconds(value);
}

template <typename NumericalRep>
uint32_t VoltageSeries<NumericalRep>::number_of_modules() const
{
    return _header.number_of_modules();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::number_of_modules(uint32_t value)
{
    _header.number_of_modules(value);
}

template <typename NumericalRep>
uint32_t VoltageSeries<NumericalRep>::number_of_samples() const
{
    return _header.number_of_samples();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::number_of_samples(uint32_t value)
{
    _header.number_of_samples(value);
}

template <typename NumericalRep>
size_t VoltageSeries<NumericalRep>::size() const
{
    return _data.size();
}

template <typename NumericalRep>
float VoltageSeries<NumericalRep>::sampling_time() const
{
    return _header.sampling_time();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::sampling_time(float value)
{
    _header.sampling_time(value);
}

template <typename NumericalRep>
uint64_t VoltageSeries<NumericalRep>::packet_sequence_number() const
{
    return _header.packet_sequence_number();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::packet_sequence_number(uint64_t value)
{
    _header.packet_sequence_number(value);
}

template <typename NumericalRep>
uint32_t VoltageSeries<NumericalRep>::packet_number() const
{
    return _header.packet_number();
}

template <typename NumericalRep>
void VoltageSeries<NumericalRep>::packet_number(uint32_t value)
{
    _header.packet_number(value);
}
