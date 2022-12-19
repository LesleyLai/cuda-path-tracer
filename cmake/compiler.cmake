# Compiler specific settings

# Link this 'library' to use the standard warnings
add_library(cuda_pt_compiler_warnings INTERFACE)
add_library(cuda_pt_compiler_options INTERFACE)

option(CUDA_PATH_TRACER_WARNING_AS_ERROR "Treats compiler warnings as errors" ON)
if (MSVC)
    target_compile_options(cuda_pt_compiler_warnings INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:
            /W4
            "/permissive-"
            /wd4201
            /wd4245
            /wd4324 # Disable "structure was padded due to alignment specifier"
            /wd4127
            >)
    target_compile_definitions(cuda_pt_compiler_warnings INTERFACE _CRT_SECURE_NO_WARNINGS)
    if (CUDA_PATH_TRACER_WARNING_AS_ERROR)
        target_compile_options(cuda_pt_compiler_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>: /WX >)
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cuda_pt_compiler_warnings
            INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:
            -Wall
            -Wextra
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wpedantic
            -Wconversion
            -Wnull-dereference
            -Wdouble-promotion
            -Wformat=2
            >)
    if (CUDA_PATH_TRACER_WARNING_AS_ERROR)
        target_compile_options(cuda_pt_compiler_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif ()

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(cuda_pt_compiler_warnings
                INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:
                -Wmisleading-indentation
                -Wduplicated-cond
                -Wduplicated-branches
                -Wlogical-op
                -Wuseless-cast
                >
                )
    endif ()
endif ()

target_compile_options(
        cuda_pt_compiler_warnings INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored
        >
)

target_compile_options(
        cuda_pt_compiler_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        --generate-line-info # For source information
        >
)

option(CUDA_PATH_TRACER_ENABLE_PCH "Enable Precompiled Headers" OFF)
if (CUDA_PATH_TRACER_ENABLE_PCH)
    target_precompile_headers(cuda_pt_compiler_warnings INTERFACE
            <algorithm>
            <array>
            <vector>
            <string>
            <utility>
            <functional>
            <memory>
            <memory_resource>
            <string_view>
            <cmath>
            <cstddef>
            <type_traits>
            )
endif ()

option(CUDA_PATH_TRACER_USE_ASAN "Enable the Address Sanitizers" OFF)
if (CUDA_PATH_TRACER_USE_ASAN)
    message("Enable Address Sanitizer")
    target_compile_options(cuda_pt_compiler_options INTERFACE
            -fsanitize=address -fno-omit-frame-pointer)
    target_link_libraries(cuda_pt_compiler_options INTERFACE
            -fsanitize=address)
endif ()

option(CUDA_PATH_TRACER_USE_TSAN "Enable the Thread Sanitizers" OFF)
if (CUDA_PATH_TRACER_USE_TSAN)
    message("Enable Thread Sanitizer")
    target_compile_options(cuda_pt_compiler_options INTERFACE
            -fsanitize=thread)
    target_link_libraries(cuda_pt_compiler_options INTERFACE
            -fsanitize=thread)
endif ()

option(CUDA_PATH_TRACER_USE_MSAN "Enable the Memory Sanitizers" OFF)
if (CUDA_PATH_TRACER_USE_MSAN)
    message("Enable Memory Sanitizer")
    target_compile_options(cuda_pt_compiler_options INTERFACE
            -fsanitize=memory -fno-omit-frame-pointer)
    target_link_libraries(cuda_pt_compiler_options INTERFACE
            -fsanitize=memory)
endif ()

option(CUDA_PATH_TRACER_USE_UBSAN "Enable the Undefined Behavior Sanitizers" OFF)
if (CUDA_PATH_TRACER_USE_UBSAN)
    message("Enable Undefined Behavior Sanitizer")
    target_compile_options(cuda_pt_compiler_options INTERFACE
            -fsanitize=undefined)
    target_link_libraries(cuda_pt_compiler_options INTERFACE
            -fsanitize=undefined)
endif ()