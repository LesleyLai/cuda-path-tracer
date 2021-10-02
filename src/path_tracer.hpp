#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include <memory>

#include "cuda_buffer.hpp"
#include "ray.hpp"

class PathTracer {
  cuda::Buffer<Ray> rays_ = nullptr;

public:
  PathTracer();

  void create_buffers(unsigned int width, unsigned int height);
  void path_trace(uchar4* PBOpos, unsigned int width, unsigned int height);
};
