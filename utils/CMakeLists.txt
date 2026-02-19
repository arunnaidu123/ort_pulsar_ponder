add_library(utils
    src/DedispHeader.cpp
    src/SendStuff.cu
    src/PinCpu.cpp
)

set_target_properties(utils PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(utils
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
)