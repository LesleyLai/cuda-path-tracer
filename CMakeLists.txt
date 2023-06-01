cmake_minimum_required(VERSION 3.17)

project(cuda_path_tracer VERSION 0.0.0 LANGUAGES CXX CUDA)

include("cmake/StandardProjectSettings.cmake")
include("cmake/compiler.cmake")

option(CUDA_PATH_TRACER_COPY_ASSETS "Copy assets to the build directory" ON)

add_subdirectory(third-party)
add_subdirectory(src)
add_subdirectory(test)