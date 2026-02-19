add_library(legacy_dedispersion
    src/LegacyDedispersion.cu
    kernels/kernels.cu
)

set_target_properties(legacy_dedispersion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(legacy_dedispersion PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(legacy_dedispersion
    PUBLIC
        ${CUDA_CUDART_LIBRARY}
)