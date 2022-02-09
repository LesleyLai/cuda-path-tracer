find_package(CUDAToolkit REQUIRED)

find_package(fmt REQUIRED)
find_package(glm REQUIRED)

add_library(common INTERFACE sphere.hpp ray.hpp span.hpp)
target_link_libraries(common INTERFACE fmt::fmt glm::glm
        cuda_pt_compiler_warnings
        cuda_pt_compiler_options)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(assimp)
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(tl-function-ref REQUIRED)

add_executable(app
        path_tracer.cu
        path_tracer.hpp
        main.cpp
        shader.hpp
        shader.cpp
        window.cpp
        window.hpp
        app.hpp
        app.cpp
        cuda_utils/cuda_check.hpp
        cuda_utils/cuda_buffer.hpp
        distributions.cuh
        preview_renderer.cpp
        preview_renderer.hpp
        gui.cpp
        gui.hpp
        camera.cpp
        camera.hpp
        hit_record.hpp
        intersections.cuh
        triangle.hpp
        scene.hpp material.hpp scene_builder.hpp mesh.hpp scene_builder.cpp
        denoising/edge_avoiding_a_trous_denoiser.cu
        denoising/edge_avoiding_a_trous_denoiser.hpp cuda_utils/2d_indices.cuh)
target_link_libraries(app PRIVATE
        common
        glfw::glfw
        glad::glad
        tl::function-ref
        assimp::assimp
        third-party::imgui_bindings
        CUDA::cudart)
set_property(TARGET app PROPERTY CUDA_ARCHITECTURES OFF)

add_custom_target(assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/bin
        )
add_dependencies(app assets)
