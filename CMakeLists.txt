cmake_minimum_required(VERSION 3.15...3.26)

project(nanobind_project LANGUAGES CXX)
# find_package(OpenMP REQUIRED)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_archive STABLE_ABI NB_STATIC src/archive.cpp)
nanobind_add_module(_reader STABLE_ABI NB_STATIC src/reader.cpp src/vtk_support.cpp)

# Compiler-specific options
if(MSVC)
  # Use MSVC optimization levels and OpenMP setup
  target_compile_options(_archive PRIVATE /O2 /std:c++17)
  target_compile_options(_reader PRIVATE /O2 /std:c++17)
else()
  # Assuming GCC or Clang
  target_compile_options(_archive PRIVATE -O3)
  target_compile_options(_reader PRIVATE -O3)
endif()

# Example debugging
# set solib-search-path /home/user/python/.venv311/lib/python3.11/site-packages/mapdl_archive/
# set breakpoint with b _reader.cpp:<LINE_NUMBER>
# target_compile_options(_reader PRIVATE -g -O0)

# Install directive for scikit-build-core
install(TARGETS _archive LIBRARY DESTINATION mapdl_archive)
install(TARGETS _reader LIBRARY DESTINATION mapdl_archive)
