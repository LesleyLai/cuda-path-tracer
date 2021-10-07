find_package(Catch2 REQUIRED)

add_executable(test span_test.cpp main.cpp)
target_link_libraries(test PRIVATE common Catch2::Catch2)