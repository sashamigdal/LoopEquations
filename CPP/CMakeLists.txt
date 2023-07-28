cmake_minimum_required(VERSION 3.24)
project("DS")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(${PROJECT_NAME} src/DS.cpp)
