cmake_minimum_required(VERSION 3.12)

project(mnist_reader)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
  message(WARNING "OpenCV not found! The dataset view function 'show_dataset' will not be available!")
endif()

set(HEADER_FILES mnist_reader.h mnist_reader_common.h ${OpenCV_INCLUDE_DIRS})

set(MNIST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${MNIST_DIR})

add_library(${PROJECT_NAME} INTERFACE)

target_sources(mnist_reader INTERFACE)

# Link OpenCV libraries to target library
target_link_libraries(${PROJECT_NAME} INTERFACE ${OpenCV_LIBS})
