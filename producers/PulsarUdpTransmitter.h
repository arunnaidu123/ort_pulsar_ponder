#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "argus/utils/RingBuffer.h"
#include "PulsarUdpPacket.h"
#include <thread>
#include <chrono>
template <typename BufferType>
class PulsarUdpTransmitter
{

public:
    PulsarUdpTransmitter(RingBuffer<BufferType>& ring,
                   const std::string& local_ip,
                   uint16_t local_port,
                   const std::string& remote_ip,
                   uint16_t remote_port);


    void run();


private:
    RingBuffer<BufferType>& _ring;
    PulsarUdpPacket<int8_t, 4096> _packet;
    int _sock;
    sockaddr_in _remote_addr;
    unsigned _count;
};

#include "detail/PulsarUdpTransmitter.cpp"