find_package(CUDAToolkit REQUIRED)

find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(assimp)
find_package(tl-function-ref REQUIRED)

add_library(cuda_pt_lib
        sphere.hpp
        ray.hpp
        span.hpp
        path_tracer.cu
        path_tracer.hpp
        cuda_utils/cuda_check.hpp
        cuda_utils/cuda_buffer.hpp
        cuda_utils/2d_indices.cuh
        camera.cpp
        camera.hpp
        distributions.cuh
        hit_record.hpp
        intersections.cuh
        triangle.hpp
        scene.hpp
        material.hpp
        scene_builder.hpp
        mesh.hpp
        scene_builder.cpp
        denoising/edge_avoiding_a_trous_denoiser.cu
        denoising/edge_avoiding_a_trous_denoiser.hpp
        )
add_library(cuda_pt::lib ALIAS cuda_pt_lib)

target_link_libraries(cuda_pt_lib PUBLIC fmt::fmt glm::glm
        cuda_pt_compiler_warnings
        cuda_pt_compiler_options
        tl::function-ref
        assimp::assimp
        CUDA::cudart)
target_include_directories(cuda_pt_lib PUBLIC ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET cuda_pt_lib PROPERTY CUDA_ARCHITECTURES OFF)

