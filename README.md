# CUDA Path Tracer

[![Windows](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Windows.yml/badge.svg)](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Windows.yml)
[![Ubuntu](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Ubuntu.yml)

WIP Path Tracer in CUDA.

## Build

You need to have a recent version of C++ compiler ([Visual Studio](https://www.visualstudio.com/)
, [GCC](https://gcc.gnu.org/), or [Clang](https://clang.llvm.org/)), CMake
3.17+, [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), and [Conan](https://conan.io/) package manager
installed.

To install conan, you need have a recent Python installed, and then you can do:

```
$ pip install conan
```

Afterwards, you can invoke CMake in command line to build the project

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ cmake --build .
```

or alternatively use your IDE's CMake integration.

## Run

The executables should be under the `bin` directory of the build directory:

- `cuda_pt` is the interactive app
- `test` runs the unit test

## Gallery

### Denoising

One sample-per-pixel output | [Edge-Avoiding À-Trous Wavelet Transform](https://jo.dreggn.org/home/2010_atrous.pdf) |
|---|---|
|![](images/1spp.png)|![](images/1spp_atrous_denoised.png)|

## Credit

This project consults the following resources and papers

- [Ray Tracing in One Weekend](https://raytracing.github.io/) book series
- [Physically Based Rendering: From Theory To Implementation 3rd edition](https://www.pbr-book.org/)
- Denoising
    - [Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)

## License

This repository is released under the MIT license, see [License](file:License) for more information.
