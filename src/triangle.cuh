#ifndef CUDA_PATH_TRACER_TRIANGLE_CUH
#define CUDA_PATH_TRACER_TRIANGLE_CUH

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
                               glm::vec3 pt2, HitRecord& record) -> bool
{
  const auto ve = r.origin;
  const auto vd = r.direction;
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
  if ((t < r.t_min) || (t > r.t_max)) return false;

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

#endif // CUDA_PATH_TRACER_TRIANGLE_CUH
