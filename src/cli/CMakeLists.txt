add_library(cuda_pt_cli
        cli.hpp
        cli.cpp
        )
add_library(cuda_pt::cli ALIAS cuda_pt_cli)

target_link_libraries(cuda_pt_cli PRIVATE
        cuda_pt::lib)
