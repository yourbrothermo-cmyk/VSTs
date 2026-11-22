add_library(voxforge_core STATIC
  src/Fft.cpp
  src/PitchTracker.cpp
  src/MelExtractor.cpp
  src/Resampler.cpp
  src/Notes.cpp
  src/PitchCorrector.cpp
  src/Psola.cpp
  src/NsfHifiGan.cpp
  src/OfflineEngine.cpp
  src/LiveEngine.cpp)
target_include_directories(voxforge_core PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(voxforge_core PUBLIC Threads::Threads)
if(MSVC)
  target_compile_definitions(voxforge_core PUBLIC _USE_MATH_DEFINES)
else()
  target_compile_options(voxforge_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(VOXFORGE_WITH_ONNX)
  find_path(ORT_INCLUDE onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
          ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session REQUIRED)
  find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
  target_include_directories(voxforge_core PUBLIC ${ORT_INCLUDE})
  target_link_libraries(voxforge_core PUBLIC ${ORT_LIB})
  target_compile_definitions(voxforge_core PUBLIC VOXFORGE_WITH_ONNX=1)
  if(VOXFORGE_WITH_COREML)
    target_compile_definitions(voxforge_core PUBLIC VOXFORGE_WITH_COREML=1)
  endif()
  message(STATUS "VoxForge: ONNX Runtime at ${ORT_LIB}")
endif()
