cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------
# Project (CUDA enabled here)
# ------------------------------------------------------------
project(ponder LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# Language standards
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# GPU architecture (Tesla K20c)
# ------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 35)

# ------------------------------------------------------------
# Default build type
# ------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------
# CUDA flags (safe for Kepler)
# ------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_35 -expt-relaxed-constexpr")

# ------------------------------------------------------------
# FetchContent
# ------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------
add_subdirectory(modules/pfb)
add_subdirectory(modules/dedispersion)
add_subdirectory(modules/dispersed_pulsar)
add_subdirectory(modules/legacy_dedispersion)
add_subdirectory(utils)
add_subdirectory(pipeline)
add_subdirectory(tests)