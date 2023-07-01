cmake_minimum_required(VERSION 3.12)

project(FFNetwork)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++17 -O3 -Ofast -march=native")

# Build types
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# Debug build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3")

# Include directory for nlohmann json library
set(JSON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/lib/nlohmann)

# Add mnist dataset module
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/mnist)

# Source files
file(GLOB SOURCES include/*.h src/*.cpp mlm/*.h main.cpp)

# Build executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${JSON_INCLUDE_DIR})

# Link libraries
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} mnist_reader ${OpenCV_LIBS})




