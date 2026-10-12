set(UNIT_TEST_SOURCES
    test_pfb.cpp
)

set(UNIT_TEST_LIBRARIES
    gtest
    gtest_main
    pfb
    utils
)

if(PONDER_WITH_CUDA)
    list(APPEND UNIT_TEST_SOURCES
        test_corner_turn.cpp
        test_gpu_pulsar_corner_turn.cpp
        test_fft.cpp
        cufft_many.cu
    )
    list(APPEND UNIT_TEST_LIBRARIES
        dedispersion
        gpu_pulsar_processing
    )
endif()

add_executable(unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(unit_tests ${UNIT_TEST_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(unit_tests)
