add_library(utils
    src/DedispHeader.cpp
    src/SendStuff.cpp
    src/PinCpu.cpp
)

target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
