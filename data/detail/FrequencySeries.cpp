#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <iostream>

template <typename NumericalRep>
FrequencySeries<NumericalRep>::FrequencySeries()
: _header(0, 0, 0)
, _data(0, 0)
{

}

template <typename NumericalRep>
FrequencySeries<NumericalRep>::FrequencySeries(unsigned number_of_modules, unsigned number_of_spectra, unsigned number_of_channels)
: _header(number_of_modules, number_of_spectra, number_of_channels)
, _data(number_of_modules*number_of_spectra*number_of_channels, {0, 0})
{
}

template <typename NumericalRep>
FrequencySeries<NumericalRep>::~FrequencySeries()
 {

 }

template <typename NumericalRep>
typename FrequencySeries<NumericalRep>::Iterator FrequencySeries<NumericalRep>::begin()
{
    return _data.begin();
}

template <typename NumericalRep>
typename FrequencySeries<NumericalRep>::ConstIterator FrequencySeries<NumericalRep>::cbegin() const
{
    return _data.begin();
}

template <typename NumericalRep>
typename FrequencySeries<NumericalRep>::Iterator FrequencySeries<NumericalRep>::end()
{
    return _data.end();
}

template <typename NumericalRep>
typename FrequencySeries<NumericalRep>::ConstIterator FrequencySeries<NumericalRep>::cend() const
{
    return _data.end();
}

template <typename NumericalRep>
uint32_t FrequencySeries<NumericalRep>::mjd_day() const
{
    return _header.mjd_day();
}

template <typename NumericalRep>
void  FrequencySeries<NumericalRep>::mjd_day(uint32_t value)
{
    _header.mjd_day(value);
}

template <typename NumericalRep>
uint32_t FrequencySeries<NumericalRep>::mjd_seconds() const
{
    return _header.mjd_seconds();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::mjd_seconds(uint32_t value)
{
    _header.mjd_seconds(value);
}

template <typename NumericalRep>
uint64_t FrequencySeries<NumericalRep>::mjd_nanoseconds() const
{
    return _header.mjd_nanoseconds();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::mjd_nanoseconds(uint64_t value)
{
    _header.mjd_nanoseconds(value);
}

template <typename NumericalRep>
uint32_t FrequencySeries<NumericalRep>::number_of_modules() const
{
    return _header.number_of_modules();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::number_of_modules(uint32_t value)
{
    _header.number_of_modules(value);
}

template <typename NumericalRep>
uint32_t FrequencySeries<NumericalRep>::number_of_spectra() const
{
    return _header.number_of_spectra();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::number_of_spectra(uint32_t value)
{
    _header.number_of_spectra(value);
}

template <typename NumericalRep>
uint32_t FrequencySeries<NumericalRep>::number_of_channels() const
{
    return _header.number_of_channels();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::number_of_channels(uint32_t value)
{
    _header.number_of_channels(value);
}

template <typename NumericalRep>
size_t FrequencySeries<NumericalRep>::size() const
{
    return _data.size();
}

template <typename NumericalRep>
float FrequencySeries<NumericalRep>::sampling_time() const
{
    return _header.sampling_time();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::sampling_time(float value)
{
    _header.sampling_time(value);
}

template <typename NumericalRep>
uint64_t FrequencySeries<NumericalRep>::packet_sequence_number() const
{
    return _header.packet_sequence_number();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::packet_sequence_number(uint64_t value)
{
    _header.packet_sequence_number(value);
}

template <typename NumericalRep>
uint32_t FrequencySeries<NumericalRep>::packet_number() const
{
    return _header.packet_number();
}

template <typename NumericalRep>
void FrequencySeries<NumericalRep>::packet_number(uint32_t value)
{
    _header.packet_number(value);
}