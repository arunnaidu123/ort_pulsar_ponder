add_library(dispersed_pulsar
    src/DispersedPulsar.cu
    kernels/kernels.cu
)

set_target_properties(dispersed_pulsar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(dispersed_pulsar PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dispersed_pulsar
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
)