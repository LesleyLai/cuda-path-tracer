#include <catch2/catch_test_macros.hpp>

#include "lib/span.hpp"

TEST_CASE("Span test")
{
  GIVEN("An int array with 5 elements")
  {
    int xs[] = {5, 4, 3, 2, 1};

    AND_GIVEN("A span construct from that array")
    {
      Span span{xs};
      THEN("Its size should be 5")
      {
        REQUIRE(span.size() == 5);
      }

      THEN("Its second element should be the second element of the array")
      {
        REQUIRE(span[1] == xs[1]);
      }

      THEN("You can iterate through it by begin() and end()")
      {
        std::size_t i = 0;
        for (int x : span) {
          REQUIRE(x == xs[i]);
          ++i;
        }
      }

      THEN("It can mutate the underlying array")
      {
        span[1] = 42;
        REQUIRE(xs[1] == 42);
      }

      THEN("It can implicitly convert to a span with const elements")
      {
        Span<const int> cspan = span;
        REQUIRE(cspan.size() == span.size());
        REQUIRE(cspan.data() == span.data());
      }
    }

    SECTION("Test pointer/size constructor")
    {
      const Span span{xs, std::size(xs)};
      const Span expected{xs};
      REQUIRE(span.size() == expected.size());
      REQUIRE(span.data() == expected.data());
    }
  }
}