#include "gui.hpp"
#include "app.hpp"

#include <imgui.h>

#include <glm/gtc/quaternion.hpp>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>

void init_imgui(GLFWwindow* window)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
  // Keyboard Controls
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad
  // Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450");

  // Load Fonts
  io.Fonts->AddFontFromFileTTF("fonts/Roboto-Medium.ttf", 20.0f);
}

void destroy_imgui()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

namespace {

void ToolTip(const char* desc, const char* shortcut = nullptr)
{
  if (ImGui::IsItemHovered()) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    if (shortcut) {
      ImGui::SameLine();
      ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8f);
      ImGui::Text("(%s)", shortcut);
      ImGui::PopStyleVar();
      ImGui::PopTextWrapPos();
    }
    ImGui::EndTooltip();
  }
}

void PushDisabled()
{
  const ImGuiContext& g = *GImGui;
  if ((g.CurrentItemFlags & ImGuiItemFlags_Disabled) == 0)
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, g.Style.Alpha * 0.6f);
  ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
}

void PopDisabled()
{
  const ImGuiContext& g = *GImGui;
  ImGui::PopItemFlag();
  if ((g.CurrentItemFlags & ImGuiItemFlags_Disabled) == 0) ImGui::PopStyleVar();
}

void draw_denoiser_options_gui(PathTracer& path_tracer, bool& enable_denoising)
{
  ImGui::Checkbox("Enable", &enable_denoising);
  if (!enable_denoising) { PushDisabled(); }

  constexpr const char* methods[] = {"Edge-Avoiding À-Trous Wavelet"};
  static int method_current = 0;

  auto& denoiser = path_tracer.atrous_denoiser;

  ImGui::Combo("Method", &method_current, methods, IM_ARRAYSIZE(methods));
  ImGui::SliderInt("Filter Size", &denoiser.filter_size, 1, 100);
  ImGui::SliderFloat("Color Weight", &denoiser.color_weight, 0.0f, 1.0f);
  ImGui::SliderFloat("Normal Weight", &denoiser.normal_weight, 0.0f, 1.0f);
  ImGui::SliderFloat("Position Weight", &denoiser.position_weight, 0.0f, 1.0f);

  if (!enable_denoising) { PopDisabled(); }
}

void draw_path_tracer_gui(PathTracer& path_tracer, bool& enable_denoising)
{
  ImGui::Text("%d iterations", path_tracer.iteration());
  ImGui::SameLine();
  if (ImGui::Button("Restart")) { path_tracer.restart(); }
  ToolTip("Restarts path tracing from iteration 1", "Space");

  ImGui::InputInt("Max iterations", &path_tracer.max_iterations);
  path_tracer.max_iterations = std::max(1, path_tracer.max_iterations);

  if (ImGui::CollapsingHeader("Denoiser")) {
    draw_denoiser_options_gui(path_tracer, enable_denoising);
  }
}

void draw_display_gui(DisplayBufferType& display_type)
{
  static constexpr const char* items[] = {"Path Tracing", "Color", "Normal",
                                          "Depth"};

  int item_current = static_cast<int>(display_type);
  ImGui::Combo("buffer", &item_current, items, IM_ARRAYSIZE(items));
  display_type = static_cast<DisplayBufferType>(item_current);
}

/// @return true if need to restart path tracer
[[nodiscard]] auto
draw_camera_gui(FirstPersonCameraController& first_person_camera_controller)
    -> bool
{
  ImGui::Text("First person camera");
  ImGui::Text("w/a/s/d: forward/left/backward/right");
  ImGui::Text("r/f: up/down");
  ImGui::Text("mouse right drag: pitch/yaw");

  ImGui::NewLine();
  ImGui::Text("Transformation:");

  bool pathtracer_restart_required = false;
  if (ImGui::Button("Reset")) {
    first_person_camera_controller.reset();
    pathtracer_restart_required = true;
  }

  if (glm::vec3 position = first_person_camera_controller.position();
      ImGui::InputFloat3("Position", &position[0])) {
    first_person_camera_controller.set_position(position);
    pathtracer_restart_required = true;
  }

  if (float fov_degree = glm::degrees(first_person_camera_controller.fov());
      ImGui::SliderFloat("Fov", &fov_degree, 10, 170)) {
    first_person_camera_controller.set_fov(glm::radians(fov_degree));
    pathtracer_restart_required = true;
  }

  ImGui::NewLine();
  ImGui::Text("Movement:");
  ImGui::SliderFloat("Speed", &first_person_camera_controller.speed, 0.001f,
                     100, "%.3f", ImGuiSliderFlags_Logarithmic);

  return pathtracer_restart_required;
}

} // anonymous namespace

void App::draw_gui()
{
  if (hide_control_panel_) return;

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Control Panel");
  ImGui::Text("~ to toggle");

  if (ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
      ImGui::BeginTabBar("MyTabBar", tab_bar_flags)) {
    if (ImGui::BeginTabItem("Path Tracer")) {
      draw_path_tracer_gui(path_tracer_, enable_denoising_);
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Camera")) {
      if (draw_camera_gui(first_person_camera_controller_)) {
        path_tracer_.restart();
      }
      ImGui::EndTabItem();
    }

    if (ImGui::BeginTabItem("Display")) {
      {
        draw_display_gui(display_type_);
      }

      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}