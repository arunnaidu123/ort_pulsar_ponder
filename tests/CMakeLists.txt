
add_executable(unit_tests
    test_pfb.cpp
#    test_dedispersion.cpp
#    test_pipeline.cpp
)

target_link_libraries(unit_tests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        pfb
        dedispersion
        dispersed_pulsar
        utils
)

target_include_directories(unit_tests PRIVATE /usr/include)
target_link_libraries(unit_tests PRIVATE fftw3)
target_compile_features(unit_tests PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(unit_tests)