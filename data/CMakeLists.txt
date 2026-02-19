add_library(data
    src/FrequencyHeader.cpp
)

target_include_directories(data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)