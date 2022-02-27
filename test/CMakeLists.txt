find_package(Catch2 REQUIRED)

add_executable(test span_test.cpp main.cpp transform_test.cpp glm_test_helper.hpp)
target_link_libraries(test PRIVATE cuda_pt::lib Catch2::Catch2)