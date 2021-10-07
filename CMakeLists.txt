cmake_minimum_required(VERSION 3.17)

project(cuda_path_tracer VERSION 0.0.0 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_BINARY_DIR}")

include("cmake/StandardProjectSettings.cmake")
include("cmake/StaticAnalyzers.cmake")

include("cmake/conan.cmake")
run_conan()

add_subdirectory(src)
add_subdirectory(test)