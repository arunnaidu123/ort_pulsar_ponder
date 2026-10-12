add_library(pulsar_udp_receptor
    src/PulsarUdpHeader.cpp
    src/PulsarUdpRecptor.cpp
)

target_include_directories(pulsar_udp_receptor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pulsar_udp_receptor PUBLIC
    utils
    data
)
