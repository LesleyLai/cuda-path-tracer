#include <catch2/catch.hpp>

#include "lib/transform.hpp"

#include "glm_test_helper.hpp"

#include <glm/gtc/matrix_transform.hpp>

TEST_CASE("Transforming Ray")
{
  const glm::vec3 origin{1, 2, 3};
  const glm::vec3 direction{1, 0, 0};
  const Ray ray{origin, 0, direction, 100};

  SECTION("Translating ray translates the origin")
  {
    const auto transform =
        Transform{glm::translate(glm::mat4{1.0}, glm::vec3(1, 1, 1))};
    const auto transformed = inverse_transform_ray(transform, ray);
    REQUIRE_THAT(transformed.origin, ApproxEqual(glm::vec3(0, 1, 2)));
    REQUIRE_THAT(transformed.direction, ApproxEqual(ray.direction));
    REQUIRE(transformed.t_min == ray.t_min);
    REQUIRE(transformed.t_max == ray.t_max);
  }

  SECTION("Scaling ray scales its origin")
  {
    const auto transform =
        Transform{glm::scale(glm::mat4{1.0}, glm::vec3(2, 2, 2))};
    const auto transformed = inverse_transform_ray(transform, ray);
    REQUIRE_THAT(transformed.origin, ApproxEqual(glm::vec3(0.5, 1, 1.5)));
    REQUIRE_THAT(transformed.direction, ApproxEqual(ray.direction));
    REQUIRE(transformed.t_min == ray.t_min);
    REQUIRE(transformed.t_max == ray.t_max);
  }

  SECTION("Rotating ray rotates the origin and direction")
  {
    const auto transform = Transform{
        glm::rotate(glm::mat4{1.0}, glm::pi<float>(), glm::vec3(0, 1, 0))};
    const auto transformed = inverse_transform_ray(transform, ray);
    REQUIRE_THAT(transformed.origin, ApproxEqual(glm::vec3(-1, 2, -3)));
    REQUIRE_THAT(transformed.direction, ApproxEqual(glm::vec3(-1, 0, 0)));
    REQUIRE(transformed.t_min == ray.t_min);
    REQUIRE(transformed.t_max == ray.t_max);
  }
}