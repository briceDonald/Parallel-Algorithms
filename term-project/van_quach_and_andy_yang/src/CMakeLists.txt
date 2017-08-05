cmake_minimum_required(VERSION 3.5)
project(src)
# CUDA PACKAGE
find_package(CUDA REQUIRED)

# COMPILE CU FILES

# SETUP FOR CPP FIES
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_library(utils lib/matrixUtils.h lib/matrixUtils.cpp)
#cuda_add_library(cuda_lib lib/decomposeLUGPU.h lib/decomposeLUGPU.cu)
# COMPILE AND LINK
cuda_add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
CUDA_ADD_CUBLAS_TO_TARGET(main)
target_link_libraries(main utils)
