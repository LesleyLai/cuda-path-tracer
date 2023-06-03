#include "glm_test_helper.hpp"
#include <catch2/catch_test_macros.hpp>

#include "lib/aabb.hpp"

TEST_CASE("AABB")
{
  SECTION("extent")
  {
    const AABB aabb{
        {1, 2, 3},
        {7, 6, 5},
    };
    REQUIRE_THAT(aabb.extent(), ApproxEqual(glm::vec3(6, 4, 2)));
  }

  SECTION("Max Extent")
  {
    const AABB aabb1{
        {1, 2, 3},
        {100, 6, 5},
    };
    REQUIRE(aabb1.max_extent() == 0);

    const AABB aabb2{
        {1, 2, 3},
        {7, 100, 5},
    };
    REQUIRE(aabb2.max_extent() == 1);

    const AABB aabb3{
        {1, 2, 3},
        {7, 6, 100},
    };
    REQUIRE(aabb3.max_extent() == 2);
  }

  SECTION("Surface Area")
  {
    const AABB aabb{
        {1, 2, 3},
        {7, 6, 5},
    };

    REQUIRE(aabb.surface_area() == 88);
  }
}