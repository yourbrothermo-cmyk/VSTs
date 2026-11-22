cmake_minimum_required(VERSION 3.22)
project(VoxForge VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "")
endif()

option(VOXFORGE_BUILD_PLUGIN "Build the JUCE AU/VST3 plugin" ON)
option(VOXFORGE_BUILD_TESTS "Build core unit tests" ON)
option(VOXFORGE_WITH_ONNX "Link ONNX Runtime for the NSF-HiFiGAN vocoder" OFF)
option(VOXFORGE_WITH_COREML "Use the CoreML execution provider (Apple Neural Engine/GPU) when ORT has it" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime install prefix (include/ + lib/)")

add_subdirectory(core)

if(VOXFORGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VOXFORGE_BUILD_PLUGIN)
  add_subdirectory(plugin)
endif()
