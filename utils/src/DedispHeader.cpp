#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include "../DedispHeader.h"

namespace ort {
namespace ponder {
namespace utils {

DedispHeader::DedispHeader(float dm)
    : _refdm(dm)
{
    _obits=sizeof(float)*8;
    _src_raj=0.0;
    _src_dej=0.0;
    _az_start=0.0;
    _za_start=0.0;
    _fch1 = 334.5;
    _foff = -1*16.0/(float)(256);
    _nchans = 256;
    _nbeams = 1;
    _ibeam = 1;
    _tstart = 61011.539583;
    _start_time = 0.0;
    _tsamp = 31.25e-9;
    _machine_id = 7;
    _telescope_id = 2;
}

DedispHeader::~DedispHeader()
{
}

void DedispHeader::tstart(double value)
{
    _tstart = value;
}

double DedispHeader::tstart()
{
    return _tstart;
}

std::vector<char> DedispHeader::read_string(std::ifstream& fpin)
{
    int len;
    fpin.read(as_bytes(len),sizeof(int));
    std::vector<char> temp(len);
    fpin.read(as_bytes(temp[0]),len);
    return temp;
}



int DedispHeader::write_header(std::ofstream& fpout)
{
    /* broadcast the header parameters to the output stream */
    send_string("HEADER_START", fpout);
    send_int("telescope_id",_telescope_id, fpout);
    send_int("machine_id",_machine_id, fpout);
    send_coords(_src_raj, _src_dej, _az_start, _za_start, fpout);

    send_int("data_type", 2, fpout);
    send_double("refdm", _refdm, fpout);
    send_double("fch1", _fch1, fpout);
    send_int("barycentric", 0, fpout);
    send_int("nchans", 1, fpout);
    send_int("nbits", 32, fpout);
    send_double ("tstart", _tstart, fpout);
    send_double("tsamp", _tsamp, fpout);
    send_int("nifs", 1, fpout);
    send_string("HEADER_END", fpout);
    return 0;
}

} // namespace utils
} // namespace ponder
} // namespace ort