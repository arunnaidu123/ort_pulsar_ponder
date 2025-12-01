#ifndef ORT_PONDER_UTILS_SENDSTUFF_H
#define ORT_PONDER_UTILS_SENDSTUFF_H

#include <iostream>
#include <fstream>
#include "AsBytes.h"

namespace ort {
namespace ponder {
namespace utils {

class SendStuff
{
public:
  /*send string*/
  int send_string(std::string text, std::ofstream& fpout);

  /*send double*/
  int send_double(std::string text, double value,  std::ofstream& fpout);

  /*send integer*/
  int send_int(std::string text, int value, std::ofstream& fpout);

  /*send coords*/
  void send_coords(double raj, double dej, double az, double za, std::ofstream& fpout);
};

} // namespace utils
} // namespace ponder
} // namespace ort

#endif //ORT_PONDER_UTILS_SENDSTUFF_H