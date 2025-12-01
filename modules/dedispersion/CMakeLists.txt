enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(dedispersion
    kernels/kernels.cu
    src/CoherentDedispersion.cu
)

set_target_properties(dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(dedispersion PRIVATE
    CUDA::cudart
    CUDA::cufft
)

target_include_directories(dedispersion PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)