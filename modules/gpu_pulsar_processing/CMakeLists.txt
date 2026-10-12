add_library(gpu_pulsar_processing
    GpuPulsarProcessing.cu
)

set_target_properties(gpu_pulsar_processing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(gpu_pulsar_processing PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(gpu_pulsar_processing
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
)
