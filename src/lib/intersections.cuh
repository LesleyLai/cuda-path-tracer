#ifndef CUDA_PATH_TRACER_INTERSECTIONS_CUH
#define CUDA_PATH_TRACER_INTERSECTIONS_CUH

#include "ray.hpp"
#include "sphere.hpp"

[[nodiscard]] __host__ __device__ auto inline ray_sphere_intersection_test(
    Ray ray, Sphere sphere, Intersection& record) -> bool
{
  const auto center = sphere.center;
  const auto radius = sphere.radius;

  const auto oc = ray.origin - center;

  const auto a = dot(ray.direction, ray.direction);
  const auto b = 2 * dot(ray.direction, oc);
  const auto c = dot(oc, oc) - radius * radius;
  const auto discrimination = b * b - 4 * a * c;

  if (discrimination < 0) { return false; }

  const auto sqrt_delta = std::sqrt(discrimination);
  const auto t1 = (-b - sqrt_delta) / (2 * a);
  const auto t2 = (-b + sqrt_delta) / (2 * a);

  auto hit_record_from_t = [&](float t) {
    record.t = t;
    record.point = ray(t);
    const auto outward_normal = (record.point - center) / radius;
    record.side = dot(ray.direction, outward_normal) < 0 ? HitFaceSide::front
                                                         : HitFaceSide::back;
    record.normal =
        record.side == HitFaceSide::front ? outward_normal : -outward_normal;
    return true;
  };

  // Get the smaller non-negative value of t1, t2
  if (t1 >= ray.t_min && t1 <= ray.t_max) { return hit_record_from_t(t1); }
  if (t2 >= ray.t_min && t2 <= ray.t_max) { return hit_record_from_t(t2); }
  return false;
}

[[nodiscard]] __host__ __device__ inline auto
triangle_normal(glm::vec3 pt0, glm::vec3 pt1, glm::vec3 pt2)
{
  return glm::normalize(glm::cross(pt1 - pt0, pt2 - pt0));
}

[[nodiscard]] __host__ __device__ inline auto
ray_triangle_intersection_test(Ray ray, glm::vec3 pt0, glm::vec3 pt1,
                               glm::vec3 pt2, Intersection& record) -> bool
{
  constexpr float EPSILON = 0.0000001;

  const glm::vec3 edge1 = pt1 - pt0;
  const glm::vec3 edge2 = pt2 - pt0;

  const glm::vec3 h = glm::cross(ray.direction, edge2);
  const float a = glm::dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return false; // This ray is parallel to this triangle.

  const float f = 1.0f / a;
  const glm::vec3 s = ray.origin - pt0;
  const float u = f * glm::dot(s, h);
  if (u < 0.0 || u > 1.0) return false;

  const glm::vec3 q = glm::cross(s, edge1);
  const float v = f * glm::dot(ray.direction, q);
  if (v < 0.0 || u + v > 1.0) return false;

  const float t = f * glm::dot(edge2, q);
  if ((t < ray.t_min) || (t > ray.t_max)) return false;

  record.t = t;
  record.point = ray(t);
  const auto outward_normal = triangle_normal(pt0, pt1, pt2);
  record.side = dot(ray.direction, outward_normal) < 0 ? HitFaceSide::front
                                                       : HitFaceSide::back;
  record.normal =
      record.side == HitFaceSide::front ? outward_normal : -outward_normal;
  record.material_id = 1;

  return true;
}

[[nodiscard]] __host__
    __device__ auto inline ray_aabb_intersection_test(Ray ray, AABB aabb)
        -> bool
{
  if (aabb.is_empty()) return false;

  const glm::vec3 t_min = (aabb.min - ray.origin) / ray.direction;
  const glm::vec3 t_max = (aabb.max - ray.origin) / ray.direction;

  const glm::vec3 real_min = glm::min(t_min, t_max);
  const glm::vec3 real_max = glm::max(t_min, t_max);

  const float minmax = std::min(std::min(real_max.x, real_max.y), real_max.z);
  const float maxmin = std::max(std::max(real_min.x, real_min.y), real_min.z);

  return minmax >= maxmin;
}

#endif // CUDA_PATH_TRACER_INTERSECTIONS_CUH
