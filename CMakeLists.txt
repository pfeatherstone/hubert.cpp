cmake_minimum_required(VERSION 3.23.0)
project(Hubert LANGUAGES C CXX)

# Helpers
set(IS_GCC      "$<CXX_COMPILER_ID:GNU>")
set(IS_CLANG    "$<CXX_COMPILER_ID:Clang>")
set(IS_MSVC     "$<CXX_COMPILER_ID:MSVC>")
set(IS_NOT_MSVC "$<NOT:${IS_MSVC}>")

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CPM_SOURCE_CACHE ${CMAKE_SOURCE_DIR}/.cache/cpm CACHE PATH "Directory to cache CPM packages" FORCE)
include(cmake/CPM.cmake)
include(CMakeDependentOption)
find_package(AVX)
find_package(NEON)

CPMAddPackage(
  NAME Eigen3
  VERSION 5.0.1
  URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
  OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF" "EIGEN_BUILD_TESTING OFF" "EIGEN_BUILD_DEMOS OFF" "EIGEN_BUILD_SPBENCH OFF")

CPMAddPackage(
  NAME Catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.15.0)

# Options
option(HUBERT_ENABLE_SANITIZERS "Build unit tests with sanitizers" OFF)

# Weights
get_filename_component(HUBERT_DATA ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/weights/hubert.dat REALPATH)

# Lib
add_library(hubert)
add_library(hubert::hubert ALIAS hubert)
target_compile_features(hubert PUBLIC cxx_std_20)
target_sources(hubert PRIVATE src/lib/hubert.cpp)
target_compile_definitions(hubert PRIVATE HUBERT_DATA="${HUBERT_DATA}")

target_include_directories(hubert
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/lib>
  PUBLIC $<INSTALL_INTERFACE:include>)

target_compile_options(hubert
  PRIVATE $<${IS_NOT_MSVC}:-Wall -Wextra -Wpedantic -Werror>
  PRIVATE $<$<BOOL:${AVX_FOUND}>:${AVX_CFLAGS}>
  PRIVATE $<$<BOOL:${NEON_FOUND}>:${NEON_CFLAGS}>
  PRIVATE $<$<CONFIG:Release>:-ffast-math>)
  
target_link_libraries(hubert PRIVATE Eigen3::Eigen)

# Example
add_executable(hubert.example src/examples/main.cpp)
target_link_libraries(hubert.example PRIVATE hubert::hubert)
target_link_options(hubert.example PRIVATE $<$<CONFIG:Release>:-s>)

# Test
add_executable(hubert.tests src/tests/hubert.cpp)
target_link_libraries(hubert.tests PRIVATE hubert::hubert Catch2::Catch2WithMain)
target_link_options(hubert.tests PRIVATE $<$<CONFIG:Release>:-s>)
get_filename_component(HUBERT_TEST_DATA ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/data REALPATH)
target_compile_definitions(hubert.tests PRIVATE HUBERT_TEST_DATA="${HUBERT_TEST_DATA}")

if (HUBERT_ENABLE_SANITIZERS)
  message(STATUS "Enabling sanitizers on unit tests")
  target_compile_options(hubert PUBLIC $<$<OR:${IS_GCC},${IS_CLANG}>:-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined>)
  target_link_options(hubert PUBLIC $<$<OR:${IS_GCC},${IS_CLANG}>:-fsanitize=address,undefined>)
endif()