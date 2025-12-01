#ifndef ORT_PONDER_UTILS_ASBYTES_H
#define ORT_PONDER_UTILS_ASBYTES_H

#include <iostream>
#include <fstream>

namespace ort {
namespace ponder {
namespace utils {

template<class T>
char* as_bytes(T& i)
{
  void* addr = &i;
  return static_cast<char *>(addr);
}

} // namespace utils
} // namespace ponder
} // namespace ort

#endif //ORT_PONDER_UTILS_ASBYTES_H