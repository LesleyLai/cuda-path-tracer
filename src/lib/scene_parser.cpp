#include "scene_parser.hpp"
#include "json.hpp"

#include <filesystem>
#include <fstream>

[[nodiscard]] auto read_scene(const Options& options) -> SceneDescription
{
  const std::filesystem::path path{options.filename};
  if (path.extension() == ".json") {
    std::ifstream file{path.c_str()};
    nlohmann::json json;
    file >> json;

    return scene_from_json(options, json);
  } else {
    throw std::runtime_error{"Unsupported file extension!"};
  }
}