add_executable(unit_tests
    test_corner_turn.cpp
    test_fft.cpp
    test_pfb.cpp
    cufft_many.cu
)

target_link_libraries(unit_tests
    gtest
    gtest_main
    dedispersion
    pfb
    utils
)

include(GoogleTest)
gtest_discover_tests(unit_tests)