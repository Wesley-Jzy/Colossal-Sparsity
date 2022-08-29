cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(spmm LANGUAGES CXX CUDA)
enable_language(CUDA)

add_executable(cublas_mm cublas_mm.cu)
target_link_libraries(cublas_mm -lcublas)