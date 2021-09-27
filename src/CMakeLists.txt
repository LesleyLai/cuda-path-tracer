find_package(CUDAToolkit REQUIRED)

find_package(fmt REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)

add_executable(app
        path_tracer.cu
        path_tracer.hpp
        main.cpp
        shader.hpp
        shader.cpp
        window.cpp
        window.hpp
        app.hpp
        app.cpp)
target_link_libraries(app PRIVATE fmt::fmt glm::glm glfw::glfw glad::glad CUDA::cudart)
set_property(TARGET app PROPERTY CUDA_ARCHITECTURES OFF)

add_custom_target(assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/bin
        )
add_dependencies(app assets)
