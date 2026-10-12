#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "../IpsUdpReceptor/RcptRingBuffer.h"
#include "IpsUdpPacket.h"
#include "../utils/MJDTime.h"

class IpsUdpReceptor
{
public:
    IpsUdpReceptor(RcptRingBuffer<int8_t>& ring_buf,
                const std::string& ip_locals,
                int cpu_id);

    ~IpsUdpReceptor();

    // non-copyable
    IpsUdpReceptor(const IpsUdpReceptor&) = delete;
    IpsUdpReceptor& operator=(const IpsUdpReceptor&) = delete;

    void run();    // blocking receive loop
    void stop();   // graceful stop

private:
    void setup_sockets();

private:
    RcptRingBuffer<int8_t>& _ring_buf;
    const std::string& _ip_local;
    int _cpu_id;
    int _sock;     // one fd per local IP
    bool _running;
    // per-socket state
    uint64_t _base_sequence_number;
    MJDTime  _base_mjd_time;
    int      _init_flag;
};
