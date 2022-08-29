cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(saxpy LANGUAGES CXX CUDA)
enable_language(CUDA)
add_executable(saxpy saxpy.cu)