# ------------------------------------------------------------
# Legacy dedispersion CUDA library
# ------------------------------------------------------------

# CUDA language already enabled at top level:
# project(... LANGUAGES CXX CUDA)

add_library(legacy_dedispersion
    kernels/kernels.cu
    src/LegacyDedispersion.cu
)

set_target_properties(legacy_dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)

# ------------------------------------------------------------
# Linking
# ------------------------------------------------------------
target_link_libraries(legacy_dedispersion PRIVATE
    cudart
    cufft
)

# ------------------------------------------------------------
# Includes
# ------------------------------------------------------------
target_include_directories(legacy_dedispersion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)