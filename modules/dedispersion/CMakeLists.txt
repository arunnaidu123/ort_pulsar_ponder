find_library(FFTW3F_LIBRARY fftw3f REQUIRED)

add_library(dedispersion
    src/CoherentDedispersion.cpp
    src/GpuCoherentDedispersion.cu
    src/GpuCoherentDedispersion8bit.cu
    kernels/kernels.cu
    kernels/corner_turn.cu
)

set_target_properties(dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(dedispersion PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dedispersion
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUFFT_LIBRARIES}
        ${FFTW3F_LIBRARY}
)
