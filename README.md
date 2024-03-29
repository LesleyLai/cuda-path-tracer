# CUDA Path Tracer

[![Windows](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Windows.yml/badge.svg)](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Windows.yml)
[![Ubuntu](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/LesleyLai/cuda-path-tracer/actions/workflows/Ubuntu.yml)

Interactive Path Tracer in CUDA.

## Features

- Sphere and triangle mesh primitives
- obj file loading
- BVH (bounding volume hierarchy) acceleration
- Edge-Avoiding À-Trous Wavelet Transform denoising

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

### Troubleshoot

#### Path-related errors in Windows

nvcc fatal : Cannot find compiler 'cl.exe' in PATH nvcc fatal : Could not set up the environment for Microsoft Visual
Studio

The way NVCC parse Windows PATH environment variable seems to be brittle. A reliable way is to clear the Path variable
and only set the location of `cl.exe` to the path. For example:

```
Path=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\
```

## Run

The executables should be under the `bin` directory of the build directory:

- `cuda_pt` is the main executable. The usage is `cuda_pt [option] <filename>`, where `<filename>` refers to scene
  files. By default, it executes in interactive mode, but with `--output <output filename>` option it runs in command
  line
  mode.
- `test` runs unit tests

## Gallery

### Denoising

 One sample-per-pixel output | [Edge-Avoiding À-Trous Wavelet Transform](https://jo.dreggn.org/home/2010_atrous.pdf) |
|-----------------------------|---------------------------------------------------------------------------------------|
| ![](images/1spp.png)        | ![](images/1spp_atrous_denoised.png)                                                  |

## Credit

This project consults the following resources and papers

- [Ray Tracing in One Weekend](https://raytracing.github.io/) book series
- [Physically Based Rendering: From Theory To Implementation 3rd edition](https://www.pbr-book.org/)
- Course slides from University of
  Pennsylvania [CIS 565 GPU Programming and Architecture](https://cis565-fall-2022.github.io/)
  and Dartmouth [CS 87 Rendering Algorithms](https://cs87-dartmouth.github.io/Fall2022/)
- Denoising
    - [Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)

### Dependencies

This project uses the following excellent third-party dependencies

#### CUDA

- CUDA SDK
- [Thrust](https://thrust.github.io/) - High level CUDA algorithm library

#### Asset

- [Assimp](https://github.com/assimp/assimp) - Asset loading
- [JSON for Modern C++](https://github.com/nlohmann/json) - Json loading
- [stb](https://github.com/nothings/stb) - Image reading/writing

#### Utilities

- [glm](glm.g-truc.net/) - Math
- [fmt](https://github.com/fmtlib/fmt) - String Formatting
- [cxxopts](https://github.com/jarro2783/cxxopts) - Command Line Parsing
- [tl-function-ref](https://github.com/TartanLlama/function_ref) - A `function_ref` implementation

#### Testing

- [Catch2](https://github.com/catchorg/Catch2) - Testing

#### Interactive App

- [glad](https://github.com/Dav1dde/glad) - OpenGL loader
- [GLFW](https://www.glfw.org/) - Window and input
- [Dear ImGui](https://github.com/ocornut/imgui) - GUI

## License

This repository is released under the MIT license, see [License](file:License) for more information.
