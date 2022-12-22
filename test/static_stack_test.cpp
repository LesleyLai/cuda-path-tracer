#include <catch2/catch.hpp>

#include "lib/static_stack.hpp"

TEST_CASE("Static Stack test")
{
  StaticStack<int, 16> stack;
  REQUIRE(stack.size() == 0);

  stack.push(10);
  REQUIRE(stack.top() == 10);
  REQUIRE(stack.size() == 1);

  stack.push(20);
  REQUIRE(stack.top() == 20);
  REQUIRE(stack.size() == 2);

  stack.pop();
  REQUIRE(stack.top() == 10);
  REQUIRE(stack.size() == 1);

  stack.pop();
  REQUIRE(stack.size() == 0);
}