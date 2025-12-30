cmake_minimum_required(VERSION 3.16)

# ------------------------------------------------------------------
# Project
# ------------------------------------------------------------------
project(ponder LANGUAGES CXX CUDA)

# ------------------------------------------------------------------
# C++ standard
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Build type
# ------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------------
# CUDA configuration (native, no FindCUDA)
# ------------------------------------------------------------------
# Let CMake find nvcc automatically. If needed, uncomment and set explicitly:
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# TEMPORARILY DISABLED (safe to enable later)
# set(CMAKE_CUDA_ARCHITECTURES 75)

# ------------------------------------------------------------------
# FetchContent
# ------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------
add_subdirectory(modules/pfb)
add_subdirectory(modules/dedispersion)
add_subdirectory(modules/dispersed_pulsar)
add_subdirectory(modules/legacy_dedispersion)
add_subdirectory(utils)
add_subdirectory(pipeline)
add_subdirectory(tests)