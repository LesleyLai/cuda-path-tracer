#ifndef CUDA_PATH_TRACER_CUDA_CHECK_HPP
#define CUDA_PATH_TRACER_CUDA_CHECK_HPP

#include <cuda_runtime_api.h>
#include <string_view>

namespace cuda {

void check_CUDA_error(std::string_view msg);

void cuda_check_impl(cudaError_t code, const char* file, int line,
                     bool abort = true);

} // namespace cuda

#define CUDA_CHECK(ans)                                                        \
  do {                                                                         \
    cuda::cuda_check_impl((ans), __FILE__, __LINE__);                          \
  } while (0)

#endif // CUDA_PATH_TRACER_CUDA_CHECK_HPP
