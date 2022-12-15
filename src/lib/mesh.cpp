#include "mesh.hpp"
#include "prelude.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

[[nodiscard]] auto load_obj(const char* filename) -> Mesh
{
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(
      filename, aiProcess_Triangulate | aiProcess_GenBoundingBoxes);
  if (!scene || !scene->HasMeshes()) {
    panic(fmt::format("Unable to load {}", filename));
  }

  const aiMesh* mesh = scene->mMeshes[0];
  std::vector<glm::vec3> positions;
  for (unsigned i = 0; i != mesh->mNumVertices; i++) {
    const aiVector3D v = mesh->mVertices[i];
    // const aiVector3D n = mesh->mNormals[i];
    // const aiVector3D t = mesh->mTextureCoords[0][i];
    positions.emplace_back(v.x, v.y, v.z);
  }

  std::vector<std::uint32_t> indices;
  for (unsigned i = 0; i != mesh->mNumFaces; i++)
    for (unsigned j = 0; j != 3; j++)
      indices.push_back(mesh->mFaces[i].mIndices[j]);

  const AABB aabb{
      .min = {mesh->mAABB.mMin.x, mesh->mAABB.mMin.y, mesh->mAABB.mMin.z},
      .max = {mesh->mAABB.mMax.x, mesh->mAABB.mMax.y, mesh->mAABB.mMax.z},
  };

  return Mesh{.positions = std::move(positions),
              .indices = std::move(indices),
              .aabb = aabb};
}