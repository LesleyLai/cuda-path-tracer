#include "scene_parser.hpp"

#include "lib/assets/json_parser.hpp"
#include "lib/prelude.hpp"

[[nodiscard]] auto read_scene(const CliConfigurations& configs,
                              const std::filesystem::path& asset_path)
    -> SceneDescription
{
  const auto& filename = configs.filename;
  const auto path = canonical(asset_path / filename);
  if (path.extension() == ".json") {
    SceneDescription scene_desc = scene_from_json(path.string());
    scene_desc.filename = filename;
    scene_desc.spp = configs.spp.has_value() ? *configs.spp : scene_desc.spp;
    return scene_desc;

  } else {
    panic(fmt::format("Unsupported file extension {}!",
                      path.extension().string()));
  }
}