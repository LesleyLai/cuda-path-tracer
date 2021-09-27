#pragma once

#include <vector_types.h>

#include <glm/glm.hpp>

#include <memory>

struct Ray {
  glm::vec3 origin;
  glm::vec3 direction;
};

class PathTracer {
  Ray* rays_ = nullptr;

public:
  PathTracer();
  ~PathTracer();
  PathTracer(const PathTracer&) = delete;
  auto operator=(const PathTracer&) & -> PathTracer& = delete;
  PathTracer(PathTracer&&) noexcept = delete;
  auto operator=(PathTracer&&) & noexcept -> PathTracer& = delete;

  void create_buffers(unsigned int width, unsigned int height);

  void path_trace(uchar4* PBOpos, unsigned int width, unsigned int height);
};
