#ifndef CUDA_PATH_TRACER_INTERSECTIONS_CUH
#define CUDA_PATH_TRACER_INTERSECTIONS_CUH

#include "ray.hpp"
#include "sphere.hpp"

[[nodiscard]] __host__ __device__ auto inline ray_sphere_intersection_test(
    Ray r, Sphere sphere, HitRecord& record) -> bool
{
  const auto center = sphere.center;
  const auto radius = sphere.radius;

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
    return true;
  };

  // Get the smaller non-negative value of t1, t2
  if (t1 >= 0) { return hit_record_from_t(t1); }
  if (t2 >= 0) { return hit_record_from_t(t2); }
  return false;
}

#endif // CUDA_PATH_TRACER_INTERSECTIONS_CUH
