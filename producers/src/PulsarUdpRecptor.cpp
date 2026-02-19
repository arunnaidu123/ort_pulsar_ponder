#include "PulsarUdpReceptor.h"
#include "PulsarUdpPacket.h"
#include "../utils/PinCpu.h"
#include "../utils/MJDTime.h"
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <errno.h>
#include <cmath>
#include <cstring>
// ------------------------------------------------------------
// Constructor / Destructor
// ------------------------------------------------------------

PulsarUdpReceptor::PulsarUdpReceptor(RcptRingBuffer<int8_t>& ring_buf,
                         const std::string& ip_local,
                         int cpu_id)
    : _ring_buf(ring_buf)
    , _ip_local(ip_local)
    , _cpu_id(cpu_id)
    , _epfd(-1)
    , _running(true)
{
    setup_sockets();
    setup_epoll();

    _base_sequence_number = 0;
    _init_flag = -1;

    std::cout << "UdpReceptor initialized with "
              << _sock << " sockets\n";
}

PulsarUdpReceptor::~PulsarUdpReceptor()
{
    if (_epfd >= 0)
        close(_epfd);

    if (_sock >= 0)
        close(_sock);
}

// ------------------------------------------------------------
// Socket setup
// ------------------------------------------------------------

void PulsarUdpReceptor::setup_sockets()
{
    int fd = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (fd < 0) {
        perror("socket");
        throw std::runtime_error("socket failed");
    }

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    int rcvbuf = 256 * 1024 * 1024;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF,
                    &rcvbuf, sizeof(rcvbuf)) < 0)
    {
        perror("setsockopt SO_RCVBUF");
        throw std::runtime_error("setsockopt failed");
    }

    sockaddr_in local{};
    local.sin_family      = AF_INET;
    local.sin_port        = htons(34345);
    local.sin_addr.s_addr = inet_addr(_ip_local.c_str());

    if (bind(fd, (sockaddr*)&local, sizeof(local)) < 0)
    {
        perror("bind");
        throw std::runtime_error("bind failed");
    }

    _sock = fd;
}


// ------------------------------------------------------------
// Epoll setup
// ------------------------------------------------------------

void PulsarUdpReceptor::setup_epoll()
{
    _epfd = epoll_create1(0);
    if (_epfd < 0) {
        perror("epoll_create1");
        throw std::runtime_error("epoll_create1 failed");
    }

    epoll_event ev{};
    ev.events   = EPOLLIN | EPOLLET;
    //ev.data.u32 = static_cast<uint32_t>(i);  // socket index

    if (epoll_ctl(_epfd, EPOLL_CTL_ADD, _sock, &ev) < 0)
    {
        perror("epoll_ctl");
        throw std::runtime_error("epoll_ctl failed");
    }
}

// ------------------------------------------------------------
// Receive loop
// ------------------------------------------------------------

void PulsarUdpReceptor::run()
{
    pin_cpu(_cpu_id);

    PulsarUdpPacket<int8_t, 4096> packet;
    epoll_event events[64];

    while (_running)
    {
        int nfds = epoll_wait(_epfd, events, 64, -1);
        if (nfds < 0)
        {
            if (errno == EINTR)
                continue;
            perror("epoll_wait");
            break;
        }

        for (int i = 0; i < nfds; ++i)
        {
            const int fd = _sock;

            // EPOLLET: drain socket fully
            while (true)
            {
                ssize_t n = recv(fd, &packet,
                                 sizeof(packet),
                                 MSG_DONTWAIT);
                if (n < 0)
                {
                    if (errno == EAGAIN || errno == EWOULDBLOCK)
                        break;
                    perror("recv");
                    _running = false;
                    break;
                }

                if (n != sizeof(packet))
                    continue;

                if (_init_flag == -1)
                {
                    _base_sequence_number =
                        packet.header().sequence_number() + 1;
                    _init_flag = 0;

                    _base_sequence_number = packet.header().sequence_number() + 1;
                    _base_mjd_time.day = packet.header().mjd_day();
                    _base_mjd_time.seconds = packet.header().mjd_seconds();
                    _base_mjd_time.nanoseconds = packet.header().mjd_nanoseconds();
                    uint64_t temp_nanoseconds = (8192-packet.header().packet_number())*4096*10;
                    _base_mjd_time.add(0, temp_nanoseconds);
                    _init_flag = 0;

                    std::cout << "Base sequence for socket "
                              << _base_sequence_number << "\n";

                }

                const uint64_t seq =
                    packet.header().sequence_number();

                if (seq < _base_sequence_number)
                {
                    continue;
                }


                //std::cout<<packet.header().sequence_number()<<" "<<packet.header().packet_number()<<"\n";
                uint64_t long_seq = (seq-_base_sequence_number)*8192 + packet.header().packet_number();
                auto writable = _ring_buf.get_writable(long_seq);
                auto buffer = writable->second;

                //std::cout<<seq<<" "<<_base_sequence_number<<" "<<packet.header().packet_number()<<" "<<long_seq<<"\n";
                if(buffer->mjd_day()==0)
                {
                    MJDTime temp_mjd(_base_mjd_time);
                    uint64_t temp_nanoseconds = (seq - _base_sequence_number) * 8192*4096*10;
                    uint32_t temp_seconds = temp_nanoseconds / 1'000'000'000;
                    temp_nanoseconds %= 1'000'000'000;
                    temp_mjd.add(temp_seconds, temp_nanoseconds);
                    buffer->mjd_day(temp_mjd.day);
                    buffer->mjd_seconds(temp_mjd.seconds);
                    buffer->mjd_nanoseconds(temp_mjd.nanoseconds);
                    buffer->packet_sequence_number(seq);
                    buffer->packet_number(0);
                }


                unsigned int location =  packet.header().packet_number()*packet.payload_size()/2;

                for(unsigned spectra=0; spectra<4; ++spectra)
                {
                    for(unsigned int i=0; i<512; ++i)
                    {
                        (*(buffer->begin()+spectra*512+i+location)).real(*(packet.begin()+spectra*1024+2*(511-i)));
                        (*(buffer->begin()+spectra*512+i+location)).imag(*(packet.begin()+spectra*1024+2*(511-i)+1));
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------
// Stop
// ------------------------------------------------------------

void PulsarUdpReceptor::stop()
{
    _running = false;
}
