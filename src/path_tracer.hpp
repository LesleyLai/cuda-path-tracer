#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include "cuda_buffer.hpp"
#include "ray.hpp"
#include "sphere.hpp"

class PathTracer {
  cuda::Buffer<Sphere> dev_spheres_;

public:
  PathTracer();

  void create_buffers();
  void path_trace(uchar4* PBOpos, unsigned int width, unsigned int height);
};
