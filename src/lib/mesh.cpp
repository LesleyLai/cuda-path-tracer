#include "mesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

[[nodiscard]] auto load_obj(const char* filename) -> GPUMesh
{
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(
      filename, aiProcess_Triangulate | aiProcess_GenBoundingBoxes);
  if (!scene || !scene->HasMeshes()) {
    throw std::runtime_error(fmt::format("Unable to load {}", filename));
  }

  const aiMesh* mesh = scene->mMeshes[0];
  thrust::host_vector<Vertex> vertices;
  for (unsigned i = 0; i != mesh->mNumVertices; i++) {
    const aiVector3D v = mesh->mVertices[i];
    // const aiVector3D n = mesh->mNormals[i];
    // const aiVector3D t = mesh->mTextureCoords[0][i];
    vertices.push_back(Vertex{{v.x, v.y, v.z}});
  }

  thrust::host_vector<std::uint32_t> indices;
  for (unsigned i = 0; i != mesh->mNumFaces; i++)
    for (unsigned j = 0; j != 3; j++)
      indices.push_back(mesh->mFaces[i].mIndices[j]);

  GPUMesh mesh_gpu;
  mesh_gpu.vertices = cuda::make_buffer<Vertex>(vertices.size());
  mesh_gpu.indices = cuda::make_buffer<std::uint32_t>(indices.size());
  mesh_gpu.indices_count = static_cast<std::uint32_t>(indices.size());

  thrust::copy(vertices.begin(), vertices.end(),
               thrust::device_pointer_cast(mesh_gpu.vertices.data()));
  thrust::copy(indices.begin(), indices.end(),
               thrust::device_pointer_cast(mesh_gpu.indices.data()));

  return mesh_gpu;
}