enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(legacy_dedispersion
    kernels/kernels.cu
    src/LegacyDedispersion.cu
)

set_target_properties(legacy_dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(legacy_dedispersion PRIVATE
    CUDA::cudart
    CUDA::cufft
)

target_include_directories(legacy_dedispersion PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)