add_library(ips_udp_receptor
    src/IpsUdpHeader.cpp
    src/IpsUdpReceptor.cpp
)

target_include_directories(ips_udp_receptor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ips_udp_receptor PUBLIC
    utils
    data
    pulsar_udp_receptor
)
