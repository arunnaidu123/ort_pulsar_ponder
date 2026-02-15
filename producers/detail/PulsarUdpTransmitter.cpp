
template <typename BufferType>
PulsarUdpTransmitter<BufferType>::PulsarUdpTransmitter(RingBuffer<BufferType>& ring,
                   const std::string& local_ip,
                   uint16_t local_port,
                   const std::string& remote_ip,
                   uint16_t remote_port)
    : _ring(ring)
    , _count(0)
{
    // Create socket
    _sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (_sock < 0)
        throw std::runtime_error("Socket creation failed");

    // -----------------------
    // Bind to LOCAL address
    // -----------------------
    sockaddr_in local_addr{};
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(local_port);

    if (inet_pton(AF_INET, local_ip.c_str(), &local_addr.sin_addr) <= 0)
        throw std::runtime_error("Invalid local IP");

    if (bind(_sock, reinterpret_cast<sockaddr*>(&local_addr), sizeof(local_addr)) < 0)
        throw std::runtime_error("Bind failed");

    // -----------------------
    // Setup REMOTE address
    // -----------------------
    std::memset(&_remote_addr, 0, sizeof(_remote_addr));
    _remote_addr.sin_family = AF_INET;
    _remote_addr.sin_port = htons(remote_port);

    if (inet_pton(AF_INET, remote_ip.c_str(), &_remote_addr.sin_addr) <= 0)
        throw std::runtime_error("Invalid remote IP");

        // Optional: enlarge send buffer
        int sndbuf = 64 * 1024 * 1024;
        setsockopt(_sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
}


template <typename BufferType>
void PulsarUdpTransmitter<BufferType>::run()
{
    using clock = std::chrono::steady_clock;
    auto next = clock::now();
    const auto period = std::chrono::nanoseconds(4096);
    while (true)
    {
        auto buf = _ring.get_readable();
        if(_count==0)
        {
            next = clock::now();
        }
        //const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(buf->data());
        //size_t total_bytes = buf->size() * sizeof(typename BufferType::value_type);

        size_t offset = 0;
        unsigned packet_count=0;
        for(unsigned int i=0; i<_ring.buffer_size(); i+=sizeof(_packet))
        {
            _packet.header().sequence_number(buf->packet_sequence_number());
            _packet.header().packet_number(buf->packet_number()+packet_count);
            _packet.header().mjd_day(buf->mjd_day());
            _packet.header().mjd_seconds(buf->mjd_seconds());
            _packet.header().mjd_nanoseconds(buf->mjd_nanoseconds());
            ++packet_count;
            std::memcpy(reinterpret_cast<int8_t*>(_packet.begin())
                       , reinterpret_cast<int8_t*>(&*(buf->begin()+i))
                       , 4096);
            ssize_t sent = sendto(_sock, &_packet, sizeof(_packet), 0, reinterpret_cast<sockaddr*>(&_remote_addr), sizeof(_remote_addr));

            if (sent < 0)
            {
                perror("sendto");
                break;
            }
            next += period;
            std::this_thread::sleep_until(next);
        }

        //if(_count%20==0) std::cout<<"UDP time: "<<buf->mjd_day()<<" "<<buf->mjd_seconds()<<" "<<buf->mjd_nanoseconds()<<"\n";
        _count++;
    }

}