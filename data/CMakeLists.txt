add_library(data
    src/VoltageHeader.cpp
)

target_include_directories(data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
