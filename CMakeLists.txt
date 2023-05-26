cmake_minimum_required(VERSION 3.12)

project(AI_test_2)

include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
  string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
  if(flag_already_set EQUAL -1)
	check_cxx_compiler_flag("${flag}" flag_supported)
	if(flag_supported)
	  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
	endif()
	unset(flag_supported CACHE)
  endif()
endfunction()

enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
enable_cxx_compiler_flag_if_supported("-std=c++17")
enable_cxx_compiler_flag_if_supported("-g3")



# # Set path to Eigen library
set(EIGEN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/lib/eigen)

include_directories(${EIGEN_INCLUDE_DIRS})

add_library(eigen INTERFACE)

target_include_directories(eigen INTERFACE ${EIGEN_INCLUDE_DIRS})

file(GLOB SOURCES include/*.h src/*.cpp main.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})


