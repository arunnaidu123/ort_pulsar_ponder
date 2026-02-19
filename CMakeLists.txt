cmake_minimum_required(VERSION 3.16)

project(ponder LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# Standards
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Tesla K20c
set(CMAKE_CUDA_ARCHITECTURES 35)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_35 -expt-relaxed-constexpr")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------
# CUDA (legacy mode for CMake 3.16)
# ------------------------------------------------------------
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Fix runtime loader (libcudart issue)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# ------------------------------------------------------------
# GoogleTest
# ------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)
enable_testing()

# ------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------
add_subdirectory(utils)
add_subdirectory(modules/pfb)
add_subdirectory(modules/dedispersion)
add_subdirectory(modules/dispersed_pulsar)
add_subdirectory(modules/legacy_dedispersion)
add_subdirectory(producers)
add_subdirectory(data)
add_subdirectory(pipeline)
add_subdirectory(tests)