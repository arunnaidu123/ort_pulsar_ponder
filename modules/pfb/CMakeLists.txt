find_library(FFTW3_LIBRARY fftw3 REQUIRED)

add_library(pfb
    src/PolyPhaseFB.cpp
)

target_include_directories(pfb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(pfb
    PUBLIC
        ${FFTW3_LIBRARY}
)