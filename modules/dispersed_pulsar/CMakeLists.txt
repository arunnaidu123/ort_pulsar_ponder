# ------------------------------------------------------------
# Dispersed pulsar CUDA library
# ------------------------------------------------------------

# CUDA language already enabled at top level:
# project(... LANGUAGES CXX CUDA)

add_library(dispersed_pulsar
    kernels/kernels.cu
    src/DispersedPulsar.cu
)

set_target_properties(dispersed_pulsar PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

# ------------------------------------------------------------
# Linking
# ------------------------------------------------------------
target_link_libraries(dispersed_pulsar PRIVATE
    cudart
    cufft
)

# ------------------------------------------------------------
# Includes
# ------------------------------------------------------------
target_include_directories(dispersed_pulsar PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)