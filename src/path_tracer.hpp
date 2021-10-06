#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_buffer.hpp"
#include "ray.hpp"
#include "sphere.hpp"

class PathTracer {
  cuda::Buffer<Sphere> dev_spheres_;
  cuda::Buffer<glm::vec3> dev_image_;

  std::size_t iteration_ = 0;

public:
  PathTracer();

  void resize_image(unsigned int width, unsigned int height);

  void create_buffers(unsigned int width, unsigned int height);
  void path_trace(uchar4* PBOpos, unsigned int width, unsigned int height);
};
