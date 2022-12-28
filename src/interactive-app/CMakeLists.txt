find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)

add_library(cuda_pt_interactive_app
        shader.hpp
        shader.cpp
        window.cpp
        window.hpp
        app.hpp
        app.cpp
        preview_renderer.cpp
        preview_renderer.hpp
        gui.cpp
        gui.hpp
        first_person_camera_controller.cpp
        first_person_camera_controller.hpp)
add_library(cuda_pt::interactive_app ALIAS cuda_pt_interactive_app)

target_link_libraries(cuda_pt_interactive_app PRIVATE
        cuda_pt::lib
        glfw::glfw
        glad::glad
        third-party::imgui_bindings)
