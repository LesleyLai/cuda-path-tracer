find_package(CUDAToolkit REQUIRED)

find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(tl-function-ref REQUIRED)
find_package(jsonformoderncpp REQUIRED)
find_package(stb REQUIRED)

include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts
        GIT_TAG v3.0.0
)
FetchContent_MakeAvailable(cxxopts)
#target_link_libraries(cuda_pt_lib vk-bootstrap::vk-bootstrap)

add_library(cuda_pt_lib
        sphere.hpp
        ray.hpp
        span.hpp
        path_tracer.cu
        path_tracer.hpp
        cuda_utils/cuda_check.hpp
        cuda_utils/cuda_buffer.hpp
        cuda_utils/indices.cuh
        camera.cpp
        camera.hpp
        distributions.cuh
        hit_record.hpp
        intersections.cuh
        triangle.hpp
        scene.hpp
        material.hpp
        scene_description.hpp
        mesh.hpp
        scene_description.cpp
        denoising/edge_avoiding_a_trous_denoiser.cu
        denoising/edge_avoiding_a_trous_denoiser.hpp
        ray_gen.cuh
        ray_gen.cu
        json.cpp
        json.hpp
        scene_parser.cpp
        scene_parser.hpp
        constant_memory.cuh
        mesh.cpp
        cuda_utils/definitions.hpp
        transform.hpp
        configurations.cpp
        configurations.hpp resolution.hpp third_party_impl.cpp prelude.cpp prelude.hpp assets.cpp assets.hpp hash.cuh)
add_library(cuda_pt::lib ALIAS cuda_pt_lib)

target_link_libraries(cuda_pt_lib PUBLIC fmt::fmt glm::glm
        cuda_pt_compiler_warnings
        cuda_pt_compiler_options
        tl::function-ref
        stb::stb
        assimp::assimp
        jsonformoderncpp::jsonformoderncpp
        cxxopts::cxxopts
        CUDA::cudart)
target_include_directories(cuda_pt_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET cuda_pt_lib PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET cuda_pt_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

