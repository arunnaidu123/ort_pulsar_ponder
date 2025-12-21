enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(dispersed_pulsar
    kernels/kernels.cu
    src/DispersedPulsar.cu
)

set_target_properties(dispersed_pulsar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(dispersed_pulsar PRIVATE
    CUDA::cudart
    CUDA::cufft
)

target_include_directories(dispersed_pulsar PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)