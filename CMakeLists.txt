cmake_minimum_required(VERSION 3.17)

project(cuda_path_tracer VERSION 0.0.0 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_BINARY_DIR}")

include("cmake/StandardProjectSettings.cmake")
include("cmake/compiler.cmake")
include("cmake/conan.cmake")
run_conan()

option(CUDA_PATH_TRACER_COPY_ASSETS "Copy assets to the build directory" ON)

add_subdirectory(third-party)
add_subdirectory(src)
add_subdirectory(test)