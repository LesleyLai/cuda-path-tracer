#ifndef CUDA_PATH_TRACER_DISTRIBUTIONS_CUH
#define CUDA_PATH_TRACER_DISTRIBUTIONS_CUH

#include <thrust/random.h>

[[nodiscard]] __device__ auto
random_in_unit_sphere(thrust::default_random_engine& rng) -> glm::vec3
{
  thrust::uniform_real_distribution<float> uni(-1, 1);
  thrust::normal_distribution<float> normal(0, 1);

  glm::vec3 p{normal(rng), normal(rng), normal(rng)};
  p = normalize(p);

  const auto c = std::cbrt(uni(rng));
  return p * c;
}

#endif // CUDA_PATH_TRACER_DISTRIBUTIONS_CUH
