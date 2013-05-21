# The name of our project is "pbt". CMakeLists files in this project can
# refer to the root source directory of the project as ${PBT_SOURCE_DIR} and
# to the root binary directory of the project as ${PBT_BINARY_DIR}.
cmake_minimum_required (VERSION 2.6)
project (pbt)

# Set a path for CMake's modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
if(NOT WIN32)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wunused")
endif(NOT WIN32)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)


SET(CMAKE_CONFIGURATION_TYPES "Release;Debug" CACHE STRING "limited configs" FORCE)
SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_VERBOSE_MAKEFILE On CACHE BOOL "" FORCE)

OPTION(PBT_VERBOSE "Prints out process messages during run-time. Please turn of if performance is required" ON)
OPTION(PBT_USE_CUDA "Enables CUDA based functions" ON)

if(PBT_USE_CUDA)
FIND_PACKAGE(CUDA)
endif(PBT_USE_CUDA)

if(PBT_VERBOSE)
    add_definitions(-DVERBOSE)
endif(PBT_VERBOSE)

FIND_PACKAGE(Armadillo REQUIRED)
Find_Package(OpenCV)

# Recurse into the "libs" and "src" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
#add_subdirectory (libs)
add_subdirectory (src)


