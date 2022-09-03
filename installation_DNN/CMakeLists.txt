cmake_minimum_required(VERSION 3.0.0)
project(example02 VERSION 0.1.0)

project(sample02)
set(OpenCV_DIR "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/opencvTutorial/install")
set(TRT_DIR "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/TensorRT/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1")

add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(/home/dat/Downloads/cudaInstallation/lib64)
link_directories(/usr/lib/x86_64-linux-gnu/lib)

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs

set(PROJECT_BINARY_DIR "/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/visual_studio_code/c++/sample02/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
 option(BUILD_SHARED_LIBRARIES "Build using shared libraries" ON)

# sampleMNIST.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/sampleInference.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/sampleUtils.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/sampleOptions.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/logger.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/getOptions.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/sampleReporting.o ../../bin/chobj/sampleMNIST/sampleMNIST/../common/sampleEngines.o
add_library(sampleInference STATIC ./common/sampleInference.cpp)
add_library(sampleUtils STATIC ./common/sampleUtils.cpp)
add_library(sampleOptions STATIC ./common/sampleOptions.cpp)
add_library(logger STATIC ./common/logger.cpp)
add_library(getOptions STATIC ./common/getOptions.cpp)
add_library(sampleReporting STATIC ./common/sampleReporting.cpp)
add_library(sampleEngines STATIC ./common/sampleEngines.cpp)

# generate_export_header(sampleInference)

# include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TRT_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/visual_studio_code/c++/sample02/common)


# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${TRT_DIR}/lib)

message(STATUS "TensorRT library status:")
message(STATUS "    libraries: ${TRT_DIR}/lib")
message(STATUS "    include path: ${TRT_DIR}/include")

message(STATUS "Project status:")
message(STATUS "   include path: ${PROJECT_SOURCE_DIR}/common")

# 4 source file needed, including .cpp .cu .h
add_executable(sample02 ${PROJECT_SOURCE_DIR}/sampleMNIST.cpp ./common/sampleInference.cpp ./common/sampleInference.h ./common/sampleUtils.cpp ./common/sampleUtils.h ./common/sampleOptions.cpp ./common/sampleOptions.h ./common/logger.cpp ./common/logger.h ./common/getOptions.cpp ./common/getOptions.h ./common/sampleReporting.cpp ./common/sampleReporting.h ./common/sampleEngines.cpp ./common/sampleEngines.h)

file(GLOB sample_h ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB sample_src ${PROJECT_SOURCE_DIR}/*.cpp)
# add_executable(sample02 ${PROJECT_SOURCE_DIR}/sampleMNIST.cpp ${sample_src} ${sample_h})
message(STATUS "${sample_h}")
target_link_libraries(sample02 "nvinfer" "nvparsers" "nvinfer_plugin" "nvonnxparser" "cudart" "cublas" "cudnn" "rt" "dl" "pthread")   #5
# target_link_libraries(sample02 ${OpenCV_LIBS})          #6
# target_link_libraries(sample02 ${CUDA_LIBRARIES})   #7
# target_link_libraries(sample02 Threads::Threads)       #8
