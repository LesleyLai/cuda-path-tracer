#include "gui.hpp"
#include "app.hpp"
#include "path_tracer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

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
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.8);
      ImGui::Text("(%s)", shortcut);
      ImGui::PopStyleVar();
      ImGui::PopTextWrapPos();
    }
    ImGui::EndTooltip();
  }
}

void draw_path_tracer_gui(PathTracer& path_tracer)
{
  ImGui::Text("%d iterations", path_tracer.iteration());
  ImGui::SameLine();
  if (ImGui::Button("Restart")) { path_tracer.restart(); }
  ToolTip("Restarts path tracing from iteration 1", "Space");

  ImGui::InputInt("Max iterations", &path_tracer.max_iterations);
  path_tracer.max_iterations = std::max(1, path_tracer.max_iterations);
}

} // anonymous namespace

void App::draw_gui()
{
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::Begin("Control Panel");
  draw_path_tracer_gui(path_tracer_);

  ImGui::Text("Camera");
  if (camera_.draw_gui()) { path_tracer_.restart(); }

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

[[nodiscard]] auto Camera::draw_gui() -> bool
{
  ImGui::Text("w/a/s/d: forward/left/backward/right");
  ImGui::Text("r/f: up/down");
  ImGui::Text("mouse right drag: pitch/yaw");

  bool camera_changed = ImGui::InputFloat3("Position", &position_[0]);
  float rotation[3] = {0, glm::degrees(pitch_), glm::degrees(yaw_)};

  if (ImGui::InputFloat3("Rotation", rotation)) {
    pitch_ = restrict_pitch(glm::radians(rotation[1]));
    yaw_ = glm::radians(rotation[2]);
    camera_changed = true;
  }

  float fov = glm::degrees(fov_);
  if (ImGui::SliderFloat("Fov", &fov, 10, 170)) {
    fov_ = glm::radians(fov);
    camera_changed = true;
  }

  return camera_changed;
}