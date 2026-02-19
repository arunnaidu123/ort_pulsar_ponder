add_library(producers
    src/PulsarUdpHeader.cpp
    src/PulsarUdpRecptor.cpp
)

target_include_directories(producers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)