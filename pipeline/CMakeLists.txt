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

target_link_libraries(sim_pulsar
    PRIVATE dispersed_pulsar
)

target_link_libraries(legacy_ponder
    PRIVATE utils legacy_dedispersion
)

target_link_libraries(acq PRIVATE fftw3)