enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
find_library(FFTW3F_LIBRARY fftw3f REQUIRED)

add_library(dedispersion
    #kernels/kernels.cu
    src/CoherentDedispersion.cpp
)

set_target_properties(dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(dedispersion PRIVATE
    ${FFTW3F_LIBRARY}
    CUDA::cudart
    CUDA::cufft
)

target_include_directories(dedispersion PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)