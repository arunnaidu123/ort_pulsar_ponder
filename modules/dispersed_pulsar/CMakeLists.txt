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

find_library(CUDA_CUFFTW_LIBRARY cufftw HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

target_link_libraries(dispersed_pulsar
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_CUFFTW_LIBRARY}
)