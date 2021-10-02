#ifndef CUDA_PATH_TRACER_CUDA_CHECK_HPP
#define CUDA_PATH_TRACER_CUDA_CHECK_HPP

#include <cuda_runtime_api.h>
#include <fmt/format.h>

inline void cuda_check_impl(cudaError_t code, const char* file, int line,
                            bool abort = true)
{
  if (code != cudaSuccess) {
    fmt::print(stderr, "CUDA error: {} {} {}\n", cudaGetErrorString(code), file,
               line);
    if (abort) exit(code);
  }
}

#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    cuda_check_impl((ans), __FILE__, __LINE__);                                \
  } while (0)

#endif // CUDA_PATH_TRACER_CUDA_CHECK_HPP
