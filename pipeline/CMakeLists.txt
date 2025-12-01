add_executable(acq
    pfb_complex.cpp
)

target_link_libraries(acq
    PRIVATE utils pfb dedispersion
)

target_link_libraries(acq PRIVATE fftw3)