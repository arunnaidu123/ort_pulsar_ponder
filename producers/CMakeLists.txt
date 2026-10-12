add_subdirectory(PulsarUdpReceptor)
add_subdirectory(IpsUdpReceptor)

add_library(producers INTERFACE)

target_link_libraries(producers INTERFACE
    pulsar_udp_receptor
    ips_udp_receptor
)
