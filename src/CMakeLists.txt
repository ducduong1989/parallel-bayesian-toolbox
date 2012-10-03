if (NOT WIN32)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-fPIC")
endif (NOT WIN32)

if(PPF_USE_CUDA)
IF(CUDA_FOUND)

        MESSAGE(STATUS "CUDA has been found")
        MESSAGE(STATUS "CUDA include dirs are: " ${CUDA_INCLUDE_DIRS} "\n")
        MESSAGE(STATUS "CUDA include dirs are: " ${CUDA_LIBRARIES} "\n")

        IF(WIN32)
        #Windows
                IF(CMAKE_SIZEOF_VOID_P EQUAL 4)
                # Regular x86
                link_directories (${CUDA_TOOLKIT_ROOT_DIR}/lib/x86)
                link_directories (${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
                ELSE (CMAKE_SIZEOF_VOID_P EQUAL 4)
                # AMD64
                link_directories (${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
                ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 4)
        ELSE(WIN32)
        #Unix / Mac
                IF(CMAKE_SIZEOF_VOID_P EQUAL 4)
                # Regular x86
                link_directories (${CUDA_TOOLKIT_ROOT_DIR}/lib/)
                ELSE (CMAKE_SIZEOF_VOID_P EQUAL 4)
                # AMD64
                link_directories (${CUDA_TOOLKIT_ROOT_DIR}/lib64)
                ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 4)
        ENDIF(WIN32)

ELSE(CUDA_FOUND)

        MESSAGE(WARNING "CUDA could not be found")
        set (PPF_USE_CUDA OFF)

ENDIF(CUDA_FOUND)
endif(PPF_USE_CUDA)


IF(ARMADILLO_FOUND)

    MESSAGE(STATUS "ARMADILLO has been found")

ELSE(ARMADILLO_FOUND)

    MESSAGE(FATAL_ERROR "ARMADILLO could not be found")

ENDIF(ARMADILLO_FOUND)

include_directories(${ARMADILLO_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src/ppf)
include_directories(${PROJECT_SOURCE_DIR}/src/ppf/noises)

#Particle Filter Library
add_subdirectory (ppf)

## Models
add_subdirectory (models)

## Test applications
add_subdirectory (test_apps)

## data generators
add_subdirectory (data_generators)


## Matlab interface if Matlab libraries are on system
FIND_PACKAGE(Matlab)
FIND_PACKAGE(Octave)
if (MATLAB_FOUND OR OCTAVE_FOUND)
  message(STATUS "MATLAB found: ${MATLAB_FOUND}")
  message(STATUS "OCTAVE found: ${OCTAVE_FOUND}")
  add_subdirectory (matlabInterface)
  else(MATLAB_FOUND OR OCTAVE_FOUND)
  message(WARNING "Neither MATLAB nor OCTAVE found")
endif (MATLAB_FOUND OR OCTAVE_FOUND)

## High precision Timer
add_subdirectory(high_precision_timer)

## Machine Information Provider Class
add_subdirectory(machine_info)

## GPU performance test application
if (CUDA_FOUND)
add_subdirectory(gpu_test)
endif (CUDA_FOUND)


add_subdirectory(adtf)


