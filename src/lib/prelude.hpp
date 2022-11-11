#ifndef CUDA_PATH_TRACER_PRELUDE_HPP
#define CUDA_PATH_TRACER_PRELUDE_HPP

// A bunch of universally useful utility functions

#include <string_view>

/**
 * @brief Dumps some error messages and terminates the program
 * @param msg The error message to output before abort
 */
[[noreturn]] void panic(std::string_view msg) noexcept;

#endif // CUDA_PATH_TRACER_PRELUDE_HPP
