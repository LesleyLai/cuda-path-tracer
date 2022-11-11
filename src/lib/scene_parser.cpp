#include "scene_parser.hpp"
#include "json.hpp"
#include "prelude.hpp"

#include <filesystem>
#include <fstream>

[[nodiscard]] auto read_scene(const CliConfigurations& configs,
                              const std::filesystem::path& asset_path)
    -> SceneDescription
{
  const auto& filename = configs.filename;
  const auto path = asset_path / filename;
  if (path.extension() == ".json") {
    std::ifstream file{path.c_str()};
    if (not file.is_open()) {
      panic(fmt::format("Cannot open file {}\n", filename));
    }
    nlohmann::json json;
    file >> json;

    SceneDescription scene_desc = scene_from_json(json);
    scene_desc.filename = filename;
    scene_desc.spp = configs.spp.has_value() ? *configs.spp : scene_desc.spp;
    return scene_desc;

  } else {
    panic(fmt::format("Unsupported file extension {}!",
                      path.extension().string()));
  }
}