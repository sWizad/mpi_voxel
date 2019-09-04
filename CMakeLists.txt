cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(research)

add_definitions(-std=c++11)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -w")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

add_executable (maingl maingl.cpp viz.cpp json11.cpp)
target_link_libraries (maingl ${OpenCV_LIBS} z glut GLEW gflags ${OPENGL_LIBRARIES})
