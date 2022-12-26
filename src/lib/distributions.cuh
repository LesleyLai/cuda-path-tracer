#ifndef CUDA_PATH_TRACER_DISTRIBUTIONS_CUH
#define CUDA_PATH_TRACER_DISTRIBUTIONS_CUH

#include <thrust/random.h>

[[nodiscard]] __device__ auto
random_in_unit_sphere(thrust::default_random_engine& rng) -> glm::vec3
{
  thrust::uniform_real_distribution<float> uni(0, 1);

  const float phi = 2.f * glm::pi<float>() * uni(rng);
  const float cos_theta = 2.f * uni(rng) - 1.f;
  const float sin_theta = sqrt(1 - cos_theta * cos_theta);

  const float x = cosf(phi) * sin_theta;
  const float y = sinf(phi) * sin_theta;
  const float z = cos_theta;
  return glm::vec3{x, y, z};
}

#endif // CUDA_PATH_TRACER_DISTRIBUTIONS_CUH
