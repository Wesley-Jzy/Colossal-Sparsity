cmake_minimum_required(VERSION 3.13)

project(SPARSITY CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

### C++ and CUDA flags.
# Requires C++11.
set(CMAKE_CXX_STANDARD 11)

### Project build.
include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(sparsity)