#ifndef CUDA_PATH_TRACER_2D_INDICES_CUH
#define CUDA_PATH_TRACER_2D_INDICES_CUH

namespace cuda {

struct Index2D {
  unsigned int x = 0;
  unsigned int y = 0;
};

[[nodiscard]] __device__ constexpr auto calculate_index_2d() -> Index2D
{
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  return Index2D{x, y};
}

} // namespace cuda

#define FLATTERN_INDEX(x, y) ((x) + ((height - (y)) * width))

#endif // CUDA_PATH_TRACER_2D_INDICES_CUH
