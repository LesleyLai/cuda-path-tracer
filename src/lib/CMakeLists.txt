find_package(CUDAToolkit REQUIRED)

find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(tl-function-ref REQUIRED)
find_package(jsonformoderncpp REQUIRED)
find_package(stb REQUIRED)
find_package(spdlog REQUIRED)

include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts
        GIT_TAG v3.0.0
)
FetchContent_MakeAvailable(cxxopts)

add_library(cuda_pt_lib
        accelerators/bvh.cpp
        accelerators/bvh.hpp
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
        intersection.hpp
        intersections.cuh
        scene.hpp
        material.hpp
        scene_description.hpp
        mesh.hpp
        scene_description.cpp
        denoising/edge_avoiding_a_trous_denoiser.cu
        denoising/edge_avoiding_a_trous_denoiser.hpp
        ray_gen.cuh
        ray_gen.cu
        assets/json_parser.cpp
        assets/json_parser.hpp
        assets/scene_parser.cpp
        assets/scene_parser.hpp
        constant_memory.cuh
        cuda_utils/definitions.hpp
        transform.hpp
        configurations.cpp
        configurations.hpp resolution.hpp third_party_impl.cpp prelude.cpp prelude.hpp assets/assets.cpp assets/assets.hpp hash.cuh aabb.hpp assets/model_loader.cpp assets/model_loader.hpp static_stack.hpp)
add_library(cuda_pt::lib ALIAS cuda_pt_lib)

target_link_libraries(cuda_pt_lib PUBLIC
        cuda_pt_compiler_warnings
        cuda_pt_compiler_options
        fmt::fmt
        spdlog::spdlog
        glm::glm
        tl::function-ref
        stb::stb
        assimp::assimp
        jsonformoderncpp::jsonformoderncpp
        cxxopts::cxxopts
        CUDA::cudart)
target_include_directories(cuda_pt_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET cuda_pt_lib PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET cuda_pt_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

