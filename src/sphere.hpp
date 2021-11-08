#ifndef CUDA_PATH_TRACER_SPHERE_HPP
#define CUDA_PATH_TRACER_SPHERE_HPP

#include <optional>

#include "hit_record.hpp"

struct Sphere {
  glm::vec3 center = {};
  float radius = 0;
};

#endif // CUDA_PATH_TRACER_SPHERE_HPP
