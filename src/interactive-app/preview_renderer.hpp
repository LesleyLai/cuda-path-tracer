#ifndef CUDA_PATH_TRACER_PREVIEW_RENDERER_HPP
#define CUDA_PATH_TRACER_PREVIEW_RENDERER_HPP

#include <glad/glad.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <tl/function_ref.hpp>

#include "lib/cuda_utils/cuda_check.hpp"
#include "lib/resolution.hpp"
#include "shader.hpp"

class PreviewRenderer {
  ShaderProgram program_{};
  GLuint preview_vao_ = 0;
  GLuint pbo_ = 0;
  cudaGraphicsResource* pbo_cuda_resource_ = nullptr;
  GLuint image_ = 0;

public:
  PreviewRenderer(Resolution resolution);
  ~PreviewRenderer();
  PreviewRenderer(const PreviewRenderer&) = delete;
  auto operator=(const PreviewRenderer&) & -> PreviewRenderer& = delete;
  PreviewRenderer(PreviewRenderer&&) noexcept = delete;
  auto operator=(PreviewRenderer&&) & noexcept -> PreviewRenderer& = delete;

  void recreate_image(Resolution resolution);
  void render(Resolution resolution);

  void map_pbo(tl::function_ref<void(uchar4*)> callback);
};

#endif // CUDA_PATH_TRACER_PREVIEW_RENDERER_HPP
