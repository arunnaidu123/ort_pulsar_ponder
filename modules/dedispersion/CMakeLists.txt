# ------------------------------------------------------------
# Dedispersion library
# ------------------------------------------------------------

# CUDA language is already enabled at top-level via:
# project(... LANGUAGES CXX CUDA)

find_library(FFTW3F_LIBRARY fftw3f REQUIRED)

add_library(dedispersion
    src/CoherentDedispersion.cpp
    src/GpuCoherentDedispersion.cu
    kernels/kernels.cu
)

set_target_properties(dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------
# Linking
# ------------------------------------------------------------
target_link_libraries(dedispersion PRIVATE
    ${FFTW3F_LIBRARY}
    cudart
    cufft
)

# ------------------------------------------------------------
# Includes
# ------------------------------------------------------------
target_include_directories(dedispersion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)