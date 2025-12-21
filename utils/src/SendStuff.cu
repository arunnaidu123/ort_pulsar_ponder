#include "../SendStuff.h"

namespace ort {
namespace ponder {
namespace utils {

//template<class T>
//char* as_bytes(T& i)
//{
//  void* addr = &i;
//  return static_cast<char *>(addr);
//}

int SendStuff::send_string(std::string text, std::ofstream& fpout)
{
    int len;
    len = (int)text.size();
    fpout.write(as_bytes(len), sizeof(int));
    fpout.write(text.c_str(), text.size());
    return 0;
}

int SendStuff::send_double(std::string text, double value,  std::ofstream& fpout)
{
    send_string(text,fpout);
    fpout.write(as_bytes(value),sizeof(double));
    return 0;
}

int SendStuff::send_int(std::string text, int value, std::ofstream& fpout)
{
    send_string(text,fpout);
    fpout.write(as_bytes(value),sizeof(int));
    return 0;
}

void SendStuff::send_coords(double raj, double dej, double az, double za,std::ofstream& fpout)
{
    if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj, fpout);
    if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej, fpout);
    if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az, fpout);
    if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za, fpout);
}

} // namespace utils
} // namespace ponder
} // namespace ort