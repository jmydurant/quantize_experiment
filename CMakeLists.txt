cmake_minimum_required(VERSION 3.22)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(Quantize_Experiment CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
-O3 \
-g -lineinfo \
--expt-relaxed-constexpr \
--expt-extended-lambda \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_89,code=sm_89 \
-Wno-deprecated-declarations \
--resource-usage \
-use_fast_math \
-DJETFIRE_ENABLED=1 \
-Xcicc --uumn -Xptxas -uumn \
-Xptxas=-v \
")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions("-DENABLE_FP8")

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        cutlass/include/
        cutlass_extensions/include/
)

add_library(process_method SHARED preprocess.cuh preprocess.cu)

