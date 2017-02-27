project(CudaWithQt)
cmake_minimum_required(VERSION 2.8)

#packages
find_package(CUDA REQUIRED)

#nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)


CUDA_ADD_EXECUTABLE(test_cuda_project ${CURRENT_HEADERS} ${CURRENT_SOURCES})
