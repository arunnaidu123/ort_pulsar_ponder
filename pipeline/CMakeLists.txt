add_executable(legacy_ponder legacy_ponder.cpp)
add_executable(pfb_complex pfb_complex.cpp)
add_executable(sim_pulsar sim_pulsar.cpp)
add_executable(process_pulsar_stream process_pulsar_stream.cpp)
add_executable(process_pulsar_stream_gpu process_pulsar_stream_gpu.cu)
add_executable(process_binary_file_gpu process_binary_file_gpu.cpp)
set_source_files_properties(process_binary_file_gpu.cpp PROPERTIES LANGUAGE CUDA)
add_executable(process_inpulsar_stream process_inpulsar_stream.cpp)
add_executable(process_pulsar_file process_pulsar_file.cpp)
add_executable(process_binary_file process_binary_file.cpp)
add_executable(write_binary write_binary.cpp)

# Deployment CPU is a Westmere Xeon E5645 (SSE4.2, no AVX).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(process_binary_file PRIVATE -O3 -march=westmere -mtune=westmere)
    target_compile_options(process_pulsar_stream PRIVATE -O3 -march=westmere -mtune=westmere)
endif()

target_link_libraries(legacy_ponder
    pfb
    dedispersion
    dispersed_pulsar
    legacy_dedispersion
    producers
    utils
    data
)

target_link_libraries(pfb_complex
    dedispersion
    pfb
    utils
)

target_link_libraries(sim_pulsar
    dispersed_pulsar
    producers
    data
    utils
)

target_link_libraries(process_pulsar_stream
    dedispersion
    dispersed_pulsar
    legacy_dedispersion
    producers
    utils
    data
)

target_link_libraries(process_pulsar_stream_gpu
    producers
    utils
    data
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUFFT_LIBRARIES}
)

set_target_properties(process_pulsar_stream_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(process_binary_file_gpu
    utils
    data
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUFFT_LIBRARIES}
)

set_target_properties(process_binary_file_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(process_inpulsar_stream
    dedispersion
    dispersed_pulsar
    legacy_dedispersion
    producers
    utils
    data
)

target_link_libraries(process_pulsar_file
    dedispersion
    dispersed_pulsar
    legacy_dedispersion
    producers
    utils
    data
)

target_link_libraries(process_binary_file
    pfb
    dedispersion
    dispersed_pulsar
    legacy_dedispersion
    producers
    utils
    data
)

target_link_libraries(write_binary
    dedispersion
    dispersed_pulsar
    legacy_dedispersion
    producers
    utils
    data
)
