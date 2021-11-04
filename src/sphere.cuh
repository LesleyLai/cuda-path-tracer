#ifndef CUDA_PATH_TRACER_SPHERE_CUH
#define CUDA_PATH_TRACER_SPHERE_CUH

#include <optional>

#include "ray.hpp"

enum HitFaceSide : std::uint8_t { front, back };

struct HitRecord {
  float t = 0;
  glm::vec3 point = {};
  glm::vec3 normal = {};
  std::size_t material_id = 0;
  HitFaceSide side = HitFaceSide::front;
};

struct Sphere {
  glm::vec3 center = {};
  float radius = 0;
  std::size_t materal_id = 0;
};

[[nodiscard]] __host__ __device__ auto inline ray_sphere_intersection_test(
    Ray r, Sphere sphere, HitRecord& record) -> bool
{
  const auto center = sphere.center;
  const auto radius = sphere.radius;
  const auto material_id = sphere.materal_id;

  const auto oc = r.origin - center;

  const auto a = dot(r.direction, r.direction);
  const auto b = 2 * dot(r.direction, oc);
  const auto c = dot(oc, oc) - radius * radius;
  const auto discrimination = b * b - 4 * a * c;

  if (discrimination < 0) { return false; }

  const auto sqrt_delta = std::sqrt(discrimination);
  const auto t1 = (-b - sqrt_delta) / (2 * a);
  const auto t2 = (-b + sqrt_delta) / (2 * a);

  auto hit_record_from_t = [&](float t) {
    record.t = t;
    record.point = r(t);
    const auto outward_normal = (record.point - center) / radius;
    record.side = dot(r.direction, outward_normal) < 0 ? HitFaceSide::front
                                                       : HitFaceSide::back;
    record.normal =
        record.side == HitFaceSide::front ? outward_normal : -outward_normal;
    record.material_id = material_id;
    return true;
  };

  // Get the smaller non-negative value of t1, t2
  if (t1 >= 0) { return hit_record_from_t(t1); }
  if (t2 >= 0) { return hit_record_from_t(t2); }
  return false;
}

struct Triangle {
  glm::vec3 pt0{};
  glm::vec3 pt1{};
  glm::vec3 pt2{};
};

[[nodiscard]] __host__ __device__ inline auto
triangle_normal(glm::vec3 pt0, glm::vec3 pt1, glm::vec3 pt2)
{
  return glm::normalize(glm::cross(pt1 - pt0, pt2 - pt0));
}

[[nodiscard]] __host__ __device__ inline auto
ray_triangle_intersection_test(Ray r, glm::vec3 pt0, glm::vec3 pt1,
                               glm::vec3 pt2, float t_min, float t_max,
                               HitRecord& record) -> bool
{
  const auto [ve, vd] = r;
  const float a = pt0.x - pt1.x;
  const float b = pt0.y - pt1.y;
  const float c = pt0.z - pt1.z;
  const float d = pt0.x - pt2.x;
  const float e = pt0.y - pt2.y;
  const float f = pt0.z - pt2.z;
  const float g = vd.x;
  const float h = vd.y;
  const float i = vd.z;
  const float j = pt0.x - ve.x;
  const float k = pt0.y - ve.y;
  const float l = pt0.z - ve.z;

  const float ei_hf = e * i - h * f;
  const float gf_di = g * f - d * i;
  const float dh_eg = d * h - e * g;
  const float ak_jb = a * k - j * b;
  const float jc_al = j * c - a * l;
  const float bl_kc = b * l - k * c;

  const float M = a * ei_hf + b * gf_di + c * dh_eg;

  // compute t
  const float t = -(f * ak_jb + e * jc_al + d * bl_kc) / M;
  if ((t < t_min) || (t > t_max)) return false;

  // compute gamma
  const float gamma = (i * ak_jb + h * jc_al + g * bl_kc) / M;
  if ((gamma < 0) || (gamma > 1)) return false;

  // compute beta
  const float beta = (j * ei_hf + k * gf_di + l * dh_eg) / M;
  if ((beta < 0) || (beta > (1 - gamma))) return false;

  record.t = t;
  record.point = r(t);
  const auto outward_normal = triangle_normal(pt0, pt1, pt2);
  record.side = dot(r.direction, outward_normal) < 0 ? HitFaceSide::front
                                                     : HitFaceSide::back;
  record.normal =
      record.side == HitFaceSide::front ? outward_normal : -outward_normal;
  record.material_id = 1;

  return true;
}

#endif // CUDA_PATH_TRACER_SPHERE_CUH
