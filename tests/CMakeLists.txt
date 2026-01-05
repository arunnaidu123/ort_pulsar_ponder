add_executable(unit_tests
    test_pfb.cpp
    test_fft.cpp
    test_corner_turn.cpp
    corner_turn.cu
    cufft_many.cu

)

target_include_directories(unit_tests
    PRIVATE
        /usr/include
)

target_link_libraries(unit_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        pfb
        dedispersion
        dispersed_pulsar
        utils
        fftw3
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
