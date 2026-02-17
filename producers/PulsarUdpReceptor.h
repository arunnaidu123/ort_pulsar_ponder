#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <sys/epoll.h>

#include "RcptRingBuffer.h"
#include "PulsarUdpPacket.h"

class PulsarUdpReceptor
{
public:
    PulsarUdpReceptor(RcptRingBuffer<uint8_t>& ring_buf,
                const std::vector<std::string>& ip_locals,
                unsigned _number_of_streams,
                unsigned samples_per_slot,
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
    RcptRingBuffer<uint8_t>& _ring_buf;
    const std::vector<std::string>& _ip_locals;
    int _cpu_id;
    std::vector<int> _socks;     // one fd per local IP
    int _epfd;
    bool _running;
    // per-socket state
    uint64_t _base_sequence_number;
    int      _init_flag;
};