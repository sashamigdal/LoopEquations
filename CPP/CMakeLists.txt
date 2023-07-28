project("DS")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
add_library(${PROJECT_NAME} DS.cpp SHARED)
