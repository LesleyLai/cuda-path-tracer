#ifndef CUDA_PATH_TRACER_HIT_RECORD_HPP
#define CUDA_PATH_TRACER_HIT_RECORD_HPP

#include <glm/vec3.hpp>

enum HitFaceSide : std::uint8_t { front, back };

struct HitRecord {
  float t = 0;
  glm::vec3 point = {};
  glm::vec3 normal = {};
  std::size_t material_id = 0;
  HitFaceSide side = HitFaceSide::front;
};

#endif // CUDA_PATH_TRACER_HIT_RECORD_HPP
