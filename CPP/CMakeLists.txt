cmake_minimum_required(VERSION 3.17)
project("Euler")
set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)  # This disables GCC extension for complex numbers
find_package(OpenMP)

# if(OpenMP_CXX_FOUND)
#     target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
# else()
#     message(WARNING "OpenMP not found")
# endif()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(SOURCES src/Euler.cpp
            # src/EigenProblem.cpp
            # src/MatrixMaker.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
