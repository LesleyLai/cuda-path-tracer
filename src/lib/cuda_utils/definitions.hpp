#ifndef CUDA_PATH_TRACER_DEFINITIONS_HPP
#define CUDA_PATH_TRACER_DEFINITIONS_HPP

#ifdef __NVCC__

#define HOST_DEVICE __host__ __device__

#else

#define HOST_DEVICE

#endif

#endif // CUDA_PATH_TRACER_DEFINITIONS_HPP
