#include "cuda_check.hpp"

#include <spdlog/spdlog.h>

namespace cuda {

void check_CUDA_error(std::string_view msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fmt::print(stderr, "Cuda error: {}: {}.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

void cuda_check_impl(cudaError_t code, const char* file, int line, bool abort)
{
  if (code != cudaSuccess) {
    fmt::print(stderr, "CUDA error: {} {} {}\n", cudaGetErrorString(code), file,
               line);
    if (abort) exit(code);
  }
}

} // namespace cuda