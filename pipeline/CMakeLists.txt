get_filename_component(CUDA_ROOT "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_ROOT "${CUDA_ROOT}" DIRECTORY)
set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")


add_executable(acq
    pfb_complex.cpp
)

add_executable(sim_pulsar
    sim_pulsar.cpp
)

add_executable(legacy_ponder
    legacy_ponder.cpp
)

target_link_libraries(acq
    PRIVATE utils pfb dedispersion
)
target_include_directories(acq PRIVATE
    ${CUDA_INCLUDE_DIR}
)

target_link_libraries(sim_pulsar
    PRIVATE dispersed_pulsar
)

target_link_libraries(legacy_ponder
    PRIVATE utils legacy_dedispersion
)

target_link_libraries(acq PRIVATE fftw3)