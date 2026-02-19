add_executable(legacy_ponder legacy_ponder.cpp)
add_executable(pfb_complex pfb_complex.cpp)
add_executable(sim_pulsar sim_pulsar.cpp)
add_executable(process_pulsar_stream process_pulsar_stream.cpp)

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