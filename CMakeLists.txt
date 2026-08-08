cmake_minimum_required(VERSION 3.27)

set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
project(mrt_local VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

# CMake 4 removed implicit compatibility with policies older than 3.5.
# SentencePiece 0.2.0 (the version pinned by Magenta) still declares 2.8.12.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum dependency policy version")

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  message(FATAL_ERROR "mrt-local MVP requires macOS on Apple Silicon")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ABSL_INTERNAL_AT_LEAST_CXX20 ON CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

include(FetchContent)

set(MAGENTART_SOURCE_DIR "" CACHE PATH "Existing magenta-realtime checkout (optional)")
if(NOT MAGENTART_SOURCE_DIR)
  FetchContent_Declare(magentart
    GIT_REPOSITORY https://github.com/magenta/magenta-realtime.git
    GIT_TAG 694a545e4ba0b88bf1150137b129582166d3e07f
    GIT_SHALLOW ON)
  FetchContent_GetProperties(magentart)
  if(NOT magentart_POPULATED)
    FetchContent_Populate(magentart)
  endif()
  set(MAGENTART_SOURCE_DIR "${magentart_SOURCE_DIR}")
endif()

# Keep dependency versions aligned with the pinned official Magenta source.
FetchContent_Declare(mlx GIT_REPOSITORY https://github.com/ml-explore/mlx.git GIT_TAG v0.31.1 GIT_SHALLOW ON)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(mlx)

FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece.git GIT_TAG v0.2.0 GIT_SHALLOW ON)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG v2.21.0 SOURCE_SUBDIR tensorflow/lite GIT_SHALLOW ON)
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(TFLITE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FETCHCONTENT_SOURCE_DIR_TENSORFLOW "${CMAKE_BINARY_DIR}/_deps/tensorflow-lite-src" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(tensorflow-lite)

add_subdirectory("${MAGENTART_SOURCE_DIR}/core" "${CMAKE_BINARY_DIR}/magentart-core")

FetchContent_Declare(httplib GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git GIT_TAG v0.30.2 GIT_SHALLOW ON)
set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_TEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

add_library(mrt_local_core
  core/engine.cpp
  core/wav.cpp)
target_include_directories(mrt_local_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(mrt_local_core PUBLIC magentart::core)

add_executable(mrt cli/main.cpp server/server.cpp)
target_link_libraries(mrt PRIVATE mrt_local_core httplib::httplib nlohmann_json::nlohmann_json)
target_compile_definitions(mrt PRIVATE
  MRT_LOCAL_VERSION="${PROJECT_VERSION}"
  MRT_LOCAL_DEFAULT_MODEL_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/models")

include(GNUInstallDirs)
install(TARGETS mrt RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

enable_testing()
add_executable(wav_test tests/wav_test.cpp core/wav.cpp)
target_include_directories(wav_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME wav_test COMMAND wav_test)
