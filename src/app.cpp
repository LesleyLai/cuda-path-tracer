#include "app.hpp"
#include "path_tracer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <bit>
#include <fmt/format.h>

namespace {

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

} // anonymous namespace

App::App()
{
  int gpu_device = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_device > device_count) {
    fmt::print(stderr, "Error: GPU device number is greater than the number of "
                       "devices! Perhaps a CUDA-capable GPU is not installed?");
    std::exit(1);
  }
  cudaDeviceProp device_prop = {};
  cudaGetDeviceProperties(&device_prop, gpu_device);

  const std::string window_title =
      fmt::format("CUDA Path Tracer [compute capability {}.{}]",
                  device_prop.major, device_prop.minor);

  if (!glfwInit()) {
    fmt::print(stderr, "Error: Cannot initialize GLFW context");
    std::exit(1);
  }
  window_ = Window(800, 800, window_title.c_str());

  glfwSetWindowUserPointer(window_.get(), this);
  glfwSetFramebufferSizeCallback(
      window_.get(), [](GLFWwindow* window, int width, int height) {
        App* app = static_cast<App*>(glfwGetWindowUserPointer(window));

        destroy_pbo(app->pbo_cuda_resource_, app->pbo_);
        app->pbo_ = create_pbo(width, height, &app->pbo_cuda_resource_);

        if (app->image_) { glDeleteTextures(1, &app->image_); }
        app->image_ = init_texture(width, height);
        app->path_tracer_.resize_image(width, height);

        glViewport(0, 0, width, height);
      });

  init_vao();

  int width = window_.width();
  int height = window_.height();
  image_ = init_texture(width, height);
  pbo_ = create_pbo(width, height, &pbo_cuda_resource_);

  program_ = ShaderBuilder{}
                 .load("shaders/pass.vert.glsl", Shader::Type::Vertex)
                 .load("shaders/pass.frag.glsl", Shader::Type::Fragment)
                 .build();
  program_.use();
  glActiveTexture(GL_TEXTURE0);

  path_tracer_.create_buffers(width, height);
}

App::~App()
{
  destroy_pbo(pbo_cuda_resource_, pbo_);
  if (image_) { glDeleteTextures(1, &image_); }
}

void App::init_vao()
{
  static constexpr GLfloat vertices[] = {
      -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
  };

  static constexpr GLfloat tex_coords[] = {1.0f, 1.0f, 0.0f, 1.0f,
                                           0.0f, 0.0f, 1.0f, 0.0f};

  static constexpr GLushort indices[] = {0, 1, 3, 3, 1, 2};

  GLuint vertex_buffer_obj_ID[3];
  glGenBuffers(3, vertex_buffer_obj_ID);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_ID[0]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(position_location_, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(position_location_);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_obj_ID[1]);
  glBufferData(GL_ARRAY_BUFFER, sizeof(tex_coords), tex_coords, GL_STATIC_DRAW);
  glVertexAttribPointer(tex_coords_location_, 2, GL_FLOAT, GL_FALSE, 0,
                        nullptr);
  glEnableVertexAttribArray(tex_coords_location_);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertex_buffer_obj_ID[2]);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);
}

void App::run_cuda()
{
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use
  // this buffer
  uchar4* dptr = nullptr;
  cudaGraphicsMapResources(1, &pbo_cuda_resource_);

  std::size_t size = 0;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(std::bit_cast<void**>(&dptr),
                                                  &size, pbo_cuda_resource_));

  path_tracer_.path_trace(dptr, window_.width(), window_.height());
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &pbo_cuda_resource_));
}

void App::render() const
{
  program_.use();
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
  glBindTexture(GL_TEXTURE_2D, image_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, window_.width(), window_.height(),
                  GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
}

void App::main_loop()
{
  while (!window_.should_close()) {
    window_.poll_events();
    run_cuda();
    render();
    window_.swap_buffers();
  }
}