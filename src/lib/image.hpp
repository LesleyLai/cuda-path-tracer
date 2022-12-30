#ifndef CUDA_PATH_TRACER_IMAGE_HPP
#define CUDA_PATH_TRACER_IMAGE_HPP

#include <string>

#include "resolution.hpp"

void write_image_file(std::string filename, Resolution res, const void* data);

#endif // CUDA_PATH_TRACER_IMAGE_HPP
