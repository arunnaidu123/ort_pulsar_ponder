#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include "../FilterbankHeader.h"

namespace ort {
namespace ponder {
namespace utils {

template<typename ValueType>
FilterbankHeader<ValueType>::FilterbankHeader()
{
    _obits=sizeof(ValueType)*8;
    _src_raj=0.0;
    _src_dej=0.0;
    _az_start=0.0;
    _za_start=0.0;
    _fch1 = 346.5;
    _foff = -1*50.0/(float)(512);
    _nchans = 512;
    _nbeams = 1;
    _ibeam = 1;
    _tstart = 61004.800000;
    _start_time = 0.0;
    _tsamp = 10.0e-9*1024;
    _machine_id = 7;
    _telescope_id = 2;
}

template<typename ValueType>
FilterbankHeader<ValueType>::~FilterbankHeader()
{
}

template<typename ValueType>
void FilterbankHeader<ValueType>::tstart(double value)
{
    _tstart = value;
}

template<typename ValueType>
double FilterbankHeader<ValueType>::tstart()
{
    return _tstart;
}

template<typename ValueType>
void FilterbankHeader<ValueType>::tsamp(double value)
{
    _tsamp = value;
}

template<typename ValueType>
double FilterbankHeader<ValueType>::tsamp()
{
    return _tsamp;
}

template<typename ValueType>
std::vector<char> FilterbankHeader<ValueType>::read_string(std::ifstream& fpin)
{
    int len;
    fpin.read(as_bytes(len),sizeof(int));
    std::vector<char> temp(len);
    fpin.read(as_bytes(temp[0]),len);
    return temp;
}

template<typename ValueType>
int FilterbankHeader<ValueType>::read_header(std::ifstream& fpin)
{
    int data_type,nifs;
    double dummy;
    std::cout<<"reading header \n";
    if(std::strcmp(read_string(fpin),"HEADER_START")==0)
    {
        std::cout<<"well looks like SIGPROC filterbank format\n";
    }
    char *temp;
    while(strcmp(temp,"HEADER_END")!=0)
    {
        temp = read_string(fpin);
        if(strcmp(temp,"machine_id")==0)
        {
            fpin.read(as_bytes(_machine_id),sizeof(int));
        }
        if(strcmp(temp,"telescope_id")==0)
        {
            fpin.read(as_bytes(_telescope_id),sizeof(int));
        }
        if(strcmp(temp,"src_raj")==0)
        {
            fpin.read(as_bytes(dummy),sizeof(double));
        }
        if(strcmp(temp,"src_dej")==0)
        {
            fpin.read(as_bytes(dummy),sizeof(double));
        }
        if(strcmp(temp,"az_start")==0)
        {
            fpin.read(as_bytes(dummy),sizeof(double));
        }
        if(strcmp(temp,"za_start")==0)
        {
            fpin.read(as_bytes(dummy),sizeof(double));
        }
        if(strcmp(temp,"data_type")==0)
        {
            fpin.read(as_bytes(data_type),sizeof(int));
        }
        if(strcmp(temp,"fch1")==0)
        {
            fpin.read(as_bytes(_fch1),sizeof(double));
        }
        if(strcmp(temp,"foff")==0)
        {
            fpin.read(as_bytes(_foff),sizeof(double));
        }
        if(strcmp(temp,"nchans")==0)
        {
            fpin.read(as_bytes(_nchans),sizeof(int));
        }
        if(strcmp(temp,"nbeams")==0)
        {
            fpin.read(as_bytes(_nbeams),sizeof(int));
        }
        if(strcmp(temp,"ibeam")==0)
        {
            fpin.read(as_bytes(_ibeam),sizeof(int));
        }
        if(strcmp(temp,"nbits")==0)
        {
            fpin.read(as_bytes(_obits),sizeof(int));
        }
        if(strcmp(temp,"tstart")==0)
        {
            fpin.read(as_bytes(_tstart),sizeof(double));
        }
        if(strcmp(temp,"tsamp")==0)
        {
            fpin.read(as_bytes(_tsamp),sizeof(double));
        }
        if(strcmp(temp,"nifs")==0)
        {
            fpin.read(as_bytes(nifs),sizeof(int));
        }
    }
    return 0;
}

template<typename ValueType>
int FilterbankHeader<ValueType>::write_header(std::ofstream& fpout)
{
    /* broadcast the header parameters to the output stream */
    send_string("HEADER_START", fpout);
    send_string("rawdatafile", fpout);
    send_string("test.dat", fpout);
    send_int("machine_id", _machine_id,fpout);
    send_int("telescope_id", _telescope_id,fpout);
    send_coords(_src_raj, _src_dej, _az_start, _za_start, fpout);
    send_int("data_type", 1, fpout);
    send_double("fch1", _fch1, fpout);
    send_double("foff", _foff, fpout);
    send_int("nchans", _nchans, fpout);

    /* beam info */
    send_int("nbeams", _nbeams, fpout);
    send_int("ibeam", _ibeam, fpout);
    /* number of bits per sample */
    send_int("nbits", _obits, fpout);
    /* start time and sample interval */
    send_double("tstart", _tstart+(double)_start_time/86400.0, fpout);
    send_double("tsamp", _tsamp, fpout);
    send_int("nifs", 1, fpout);
    send_string("HEADER_END", fpout);
    return 0;
}

} // namespace utils
} // namespace ponder
} // namespace ort