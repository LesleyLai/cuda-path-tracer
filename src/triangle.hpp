#ifndef CUDA_PATH_TRACER_TRIANGLE_HPP
#define CUDA_PATH_TRACER_TRIANGLE_HPP

#include <glm/glm.hpp>

struct Triangle {
  glm::vec3 pt0{};
  glm::vec3 pt1{};
  glm::vec3 pt2{};
};

#endif // CUDA_PATH_TRACER_TRIANGLE_HPP
