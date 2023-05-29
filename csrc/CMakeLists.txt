cmake_minimum_required(VERSION 2.8)
project(c_utils)
set (CMAKE_CXX_STANDARD 14)

# find_package(Eigen3 REQUIRED)

# include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(./eigen)

find_package(OpenMP)
if (OPENMP_FOUND)
    message(STATUS "Found OpenMP")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(pybind11)

pybind11_add_module(c_utils c_utils.cc)
