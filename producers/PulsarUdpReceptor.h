#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <sys/epoll.h>

#include "RcptRingBuffer.h"
#include "PulsarUdpPacket.h"
#include "../utils/MJDTime.h"

class PulsarUdpReceptor
{
public:
    PulsarUdpReceptor(RcptRingBuffer<int8_t>& ring_buf,
                const std::string& ip_locals,
                int cpu_id);

    ~PulsarUdpReceptor();

    // non-copyable
    PulsarUdpReceptor(const PulsarUdpReceptor&) = delete;
    PulsarUdpReceptor& operator=(const PulsarUdpReceptor&) = delete;

    void run();    // blocking receive loop
    void stop();   // graceful stop

private:
    void setup_sockets();
    void setup_epoll();

private:
    RcptRingBuffer<int8_t>& _ring_buf;
    const std::string& _ip_local;
    int _cpu_id;
    int _sock;     // one fd per local IP
    int _epfd;
    bool _running;
    // per-socket state
    uint64_t _base_sequence_number;
    MJDTime  _base_mjd_time;
    int      _init_flag;
};