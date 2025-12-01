cmake_minimum_required(VERSION 3.18)

project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# TEMPORARILY REMOVE ARCH FLAGS
# (CMake will warn, but won't error)
# set(CMAKE_CUDA_ARCHITECTURES 75)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(modules/pfb)
add_subdirectory(modules/dedispersion)
add_subdirectory(utils)
add_subdirectory(pipeline)