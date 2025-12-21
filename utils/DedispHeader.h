#ifndef ORT_PONDER_UTILS_DEDISPHEADER_H
#define ORT_PONDER_UTILS_DEDISPHEADER_H

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>
#include "SendStuff.h"

namespace ort {
namespace ponder {
namespace utils {

class DedispHeader : public SendStuff
{

public:
    DedispHeader(float dm);
    ~DedispHeader();

    void tstart(double value);
    double tstart();

    std::vector<char> read_string(std::ifstream& fpin);

    int read_header(std::ifstream& fpin);

    int write_header(std::ofstream& fpout);


private:
    float _refdm;
    int _obits;
    double _src_raj, _src_dej, _az_start, _za_start;
    double _fch1;
    double _foff;
    int _nchans;
    int _nbeams;
    int _ibeam;
    double _tstart;
    double _start_time;
    double _tsamp;
    int _machine_id;
    int _telescope_id;
    int _nacc;
};

} // namespace utils
} // namespace ponder
} // namespace ort

#endif //ORT_PONDER_UTILS_DEDISPHEADER_H