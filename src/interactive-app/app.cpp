#include "app.hpp"

#include "../lib/scene_parser.hpp"
#include "gui.hpp"
#include "preview_renderer.hpp"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <fmt/format.h>

#include <glm/gtx/compatibility.hpp>

#include <chrono>

App::App(const Options& options)
    : path_tracer_{}, first_person_camera_controller_{camera_}
{
  int gpu_device = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpu_device > device_count) {
    fmt::print(stderr, "Error: GPU device number is greater than the number of "
                       "devices! Perhaps a CUDA-capable GPU is not installed?");
    std::exit(1);
  }

  if (!glfwInit()) {
    fmt::print(stderr, "Error: Cannot initialize GLFW context");
    std::exit(1);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  window_ = Window(800, 800, "CUDA Path Tracer");

  glfwSetWindowUserPointer(window_.get(), this);
  glfwSetFramebufferSizeCallback(
      window_.get(), [](GLFWwindow* window, int width, int height) {
        auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));

        const Resolution res{width, height};

        app->preview_->recreate_image(res);
        app->path_tracer_.resize_image(res.to_unsigned());

        glViewport(0, 0, width, height);
      });
  glfwSetKeyCallback(window_.get(), [](GLFWwindow* window, int key,
                                       int /*scancode*/, int action, int mods) {
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    switch (action) {
    case GLFW_PRESS:
      switch (key) {
      case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, GL_TRUE); break;
      case GLFW_KEY_SPACE: app->path_tracer_.restart(); break;
      case GLFW_KEY_GRAVE_ACCENT: {
        if (mods == GLFW_MOD_SHIFT)
          app->hide_control_panel_ = !app->hide_control_panel_; // ~
      } break;
      default: break;
      }
      break;
    case GLFW_REPEAT: {
      switch (key) {
      case GLFW_KEY_W:
        app->first_person_camera_controller_.move(
            FirstPersonCameraController::MoveDirection::forward);
        app->path_tracer_.restart();
        break;
      case GLFW_KEY_A:
        app->first_person_camera_controller_.move(
            FirstPersonCameraController::MoveDirection::left);
        app->path_tracer_.restart();
        break;
      case GLFW_KEY_S:
        app->first_person_camera_controller_.move(
            FirstPersonCameraController::MoveDirection::backward);
        app->path_tracer_.restart();
        break;
      case GLFW_KEY_D:
        app->first_person_camera_controller_.move(
            FirstPersonCameraController::MoveDirection::right);
        app->path_tracer_.restart();
        break;
      case GLFW_KEY_R:
        app->first_person_camera_controller_.move(
            FirstPersonCameraController::MoveDirection::up);
        app->path_tracer_.restart();
        break;
      case GLFW_KEY_F:
        app->first_person_camera_controller_.move(
            FirstPersonCameraController::MoveDirection::down);
        app->path_tracer_.restart();
        break;
      default: break;
      }
    }
    default: break;
    }
  });

  glfwSetMouseButtonCallback(window_.get(), [](GLFWwindow* window, int button,
                                               int action, int /*mods*/) {
    auto* app_ptr = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
      switch (action) {
      case GLFW_PRESS: app_ptr->is_right_clicking_ = true; break;
      case GLFW_RELEASE: app_ptr->is_right_clicking_ = false;
      default: break;
      }
    }
  });

  glfwSetCursorPosCallback(
      window_.get(), [](GLFWwindow* window, double x, double y) {
        auto* app_ptr = static_cast<App*>(glfwGetWindowUserPointer(window));
        auto& first_person_camera = app_ptr->first_person_camera_controller_;

        int width = 0, height = 0;
        glfwGetWindowSize(window, &width, &height);
        const auto f_width = static_cast<float>(width);
        const auto f_height = static_cast<float>(height);

        static bool first_mouse = true;
        static float last_x = f_width / 2.0f;
        static float last_y = f_height / 2.0f;
        if (first_mouse) {
          last_x = static_cast<float>(x);
          last_y = static_cast<float>(y);
          first_mouse = false;

          // TODO: update rotation base
        }

        // reversed since y-coordinates go from bottom to top
        const auto x_offset = static_cast<float>(x) - last_x;
        const auto y_offset = last_y - static_cast<float>(y);

        last_x = static_cast<float>(x);
        last_y = static_cast<float>(y);

        if (app_ptr->is_right_clicking_) {
          first_person_camera.mouse_move(glm::radians(x_offset),
                                         glm::radians(y_offset));
          app_ptr->path_tracer_.restart();
        }
      });

  glfwSetErrorCallback([](int error, const char* description) {
    fmt::print(stderr, "Error {}: {}\n", error, description);
    std::fflush(stderr);
  });

  if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    fmt::print(stderr, "Failed to initialize OpenGL context");
    std::exit(1);
  }

  const auto resolution = window_.resolution();
  preview_ = std::make_unique<PreviewRenderer>(resolution);

  const SceneDescription scene_desc = read_scene(options);
  //  SceneDescription scene_desc;
  //  scene_desc.add_material("ground", DiffuseMateral{glm::vec3(0.5, 0.7,
  //  0.0)}); scene_desc.add_object(
  //      Sphere{.center = glm::vec3{}, .radius = 1000},
  //      Transform{glm::translate(glm::mat4{1.0}, glm::vec3(0, -1004, 0))},
  //      "ground");
  //
  //  static constexpr float radius = 0.5f;
  //  static constexpr int num_s = 40;
  //  static constexpr int num_t = 40;

  /*
  (define radius 0.5)
  (define s-count 40)
  (define t-count 40)
  (iter (cartesian-product (range 0 s-count) (range 0 t-count))
    (lambda (is it)
      (let [(s (/ (+ is 0.5) s-count))
            (t (/ (+ it 0.5) t-count))
            (u (- (* s 8) 4))
            (v (* t 6.25))]
    ))

  */

  //  for (int is = 0; is < num_s; ++is) {
  //    for (int it = 0; it < num_t; ++it) {
  //      const float s = (is + 0.5f) / num_s;
  //      const float t = (it + 0.5f) / num_t;
  //      const float u = s * 8 - 4.0f;
  //      const float v = t * 6.25f;
  //      glm::vec3 center(-u * cos(v), v * cos(u) * 0.75f, u * sin(v));
  //      glm::vec3 kd = 0.3f * glm::mix(glm::mix(glm::vec3(0.9f, 0.0f, 0.0f),
  //                                              glm::vec3(0.0f, 0.9f, 0.0f),
  //                                              t),
  //                                     glm::mix(glm::vec3(0.0f, 0.0f, 0.9f),
  //                                              glm::vec3(0.0f, 0.0f, 0.0f),
  //                                              t),
  //                                     s);
  //
  //      const std::string material_name = fmt::format("{}-{}", is, it);
  //      scene_desc.add_material(material_name, DiffuseMateral{kd});
  //      scene_desc.add_object(Sphere{.center = glm::vec3{}, .radius = radius},
  //                            Transform{glm::translate(glm::mat4{1.0},
  //                            center)}, material_name);
  //    }
  //  }

  // first_person_camera_controller_.set_position(glm::vec3(10, 10, 0));

  path_tracer_.create_buffers(resolution.to_unsigned(), scene_desc);

  init_imgui(window_.get());
}

App::~App()
{
  destroy_imgui();
}

void App::run_cuda()
{
  const auto resolution = window_.u_resolution();

  using namespace std::chrono_literals;
  using Clock = std::chrono::system_clock;
  const auto start_time = Clock::now();
  do {
    path_tracer_.path_trace(camera_, resolution);

    if (enable_denoising_) { path_tracer_.denoise(resolution); }

    CUDA_CHECK(cudaDeviceSynchronize());
  } while (Clock::now() - start_time < 16ms);

  preview_->map_pbo([&](uchar4* dev_pbo) {
    path_tracer_.send_to_preview(dev_pbo, resolution);
  });
}

void App::main_loop()
{
  while (!window_.should_close()) {
    window_.poll_events();
    run_cuda();
    preview_->render(window_.resolution());
    draw_gui();
    window_.swap_buffers();
  }
}
