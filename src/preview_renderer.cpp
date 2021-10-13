#include "preview_renderer.hpp"

namespace {

static constexpr GLuint position_location = 0;
static constexpr GLuint tex_coords_location = 1;

[[nodiscard]] auto init_texture(int width, int height) -> GLuint
{
  GLuint image = 0;
  glGenTextures(1, &image);
  glBindTexture(GL_TEXTURE_2D, image);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
               GL_UNSIGNED_BYTE, nullptr);
  return image;
}

[[nodiscard]] auto create_pbo(int width, int height,
                              cudaGraphicsResource** pbo_cuda_resource)
    -> GLuint
{
  GLuint pbo = 0;

  // set up vertex data parameter
  const auto texels_count = width * height;
  const auto values_count = texels_count * 4;
  const auto size_tex_data =
      static_cast<GLsizeiptr>(sizeof(GLubyte) * values_count);

  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  glGenBuffers(1, &pbo);
  // Make this the current UNPACK buffer (OpenGL is state-based)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  // Allocate data for the buffer. 4-channel 8-bit image
  glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, nullptr, GL_DYNAMIC_COPY);
  cudaGraphicsGLRegisterBuffer(pbo_cuda_resource, pbo,
                               cudaGraphicsRegisterFlagsWriteDiscard);

  return pbo;
}

void destroy_pbo(cudaGraphicsResource* pbo_cuda_resource, GLuint pbo)
{
  if (pbo != 0) {
    cudaGraphicsUnregisterResource(pbo_cuda_resource);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glDeleteBuffers(1, &pbo);
  }
}

void init_vao(GLuint& preview_vao)
{
  static constexpr GLfloat vertices[] = {
      -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
  };

  static constexpr GLfloat tex_coords[] = {1.0f, 1.0f, 0.0f, 1.0f,
                                           0.0f, 0.0f, 1.0f, 0.0f};

  static constexpr GLushort indices[] = {0, 1, 3, 3, 1, 2};

  glGenVertexArrays(1, &preview_vao);
  glBindVertexArray(preview_vao);

  GLuint buffers[3];
  glGenBuffers(3, buffers);

  glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(position_location, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(position_location);

  glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords), tex_coords, GL_STATIC_DRAW);
  glVertexAttribPointer(tex_coords_location, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(tex_coords_location);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
}

} // anonymous namespace

void PreviewRenderer::recreate_image(int width, int height)
{
  destroy_pbo(pbo_cuda_resource_, pbo_);
  pbo_ = create_pbo(width, height, &pbo_cuda_resource_);

  if (image_) { glDeleteTextures(1, &image_); }
  image_ = init_texture(width, height);
}

PreviewRenderer::PreviewRenderer(int width, int height)
{
  program_ = ShaderBuilder{}
                 .load("shaders/pass.vert.glsl", Shader::Type::Vertex)
                 .load("shaders/pass.frag.glsl", Shader::Type::Fragment)
                 .build();
  program_.use();
  glActiveTexture(GL_TEXTURE0);

  pbo_ = create_pbo(width, height, &pbo_cuda_resource_);
  image_ = init_texture(width, height);

  init_vao(preview_vao_);
}

PreviewRenderer::~PreviewRenderer()
{
  destroy_pbo(pbo_cuda_resource_, pbo_);
  if (image_) { glDeleteTextures(1, &image_); }
}
void PreviewRenderer::render(int width, int height)
{
  program_.use();
  glBindVertexArray(preview_vao_);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, image_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                  GL_UNSIGNED_BYTE, nullptr);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

  glUseProgram(0);
  glBindVertexArray(0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void PreviewRenderer::map_pbo(tl::function_ref<void(uchar4*)> callback)
{
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
  // use this buffer
  uchar4* dev_pbo = nullptr;
  cudaGraphicsMapResources(1, &pbo_cuda_resource_);

  std::size_t size = 0;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void**>(&dev_pbo), &size, pbo_cuda_resource_));

  callback(dev_pbo);

  CUDA_CHECK(cudaGraphicsUnmapResources(1, &pbo_cuda_resource_));
}
