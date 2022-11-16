#ifndef CUDA_PATH_TRACER_HASH_CUH
#define CUDA_PATH_TRACER_HASH_CUH

[[nodiscard]] __host__ __device__ constexpr auto hash(unsigned int a)
    -> unsigned int
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

#endif // CUDA_PATH_TRACER_HASH_CUH
