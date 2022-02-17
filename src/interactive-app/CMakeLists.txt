find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)

add_executable(cuda_pt
        main.cpp
        shader.hpp
        shader.cpp
        window.cpp
        window.hpp
        app.hpp
        app.cpp
        preview_renderer.cpp
        preview_renderer.hpp
        gui.cpp
        gui.hpp)
target_link_libraries(cuda_pt PRIVATE
        cuda_pt::lib
        glfw::glfw
        glad::glad
        third-party::imgui_bindings)

add_custom_target(assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/bin
        )
add_dependencies(cuda_pt assets)