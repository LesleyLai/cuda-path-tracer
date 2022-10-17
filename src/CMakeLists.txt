add_subdirectory(lib)
add_subdirectory(interactive-app)

add_executable(cuda_pt
        main.cpp)

target_link_libraries(cuda_pt PUBLIC cuda_pt::lib cuda_pt::interactive_app)

if (CUDA_PATH_TRACER_COPY_ASSETS)
    add_custom_target(assets
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PROJECT_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/bin)
    add_dependencies(cuda_pt assets)
endif ()