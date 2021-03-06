SET(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${PROJECT_SOURCE_DIR}/src/pbt)
include_directories(${PROJECT_SOURCE_DIR}/src/models)

if (MATLAB_FOUND)
	include_directories(${MATLAB_INCLUDE_DIR})
endif (MATLAB_FOUND)

if (OCTAVE_FOUND)
	include_directories(${OCTAVE_INCLUDE_DIR})
	include_directories(${OCTAVE_INCLUDE_PATHS})
endif (OCTAVE_FOUND)

ADD_LIBRARY(matlabinterface matlab_interface.cpp matlab_interface.h)

SET(MODELS	model_fancy
			model_sinus
			model_linear
			model_wpam_3d_trajectory
      model_straight3d
			model_car_localization)

if (MATLAB_FOUND)
        MATLAB_MEX_CREATE(pbtmex pbtmex.cpp ${MATLAB_LIBRARIES} ${MODELS} pbtlib matlabinterface)
        SET_TARGET_PROPERTIES(pbtmex PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/matlab")
endif (MATLAB_FOUND)

if (OCTAVE_FOUND)
        OCTAVE_MEX_CREATE(pbtoct pbtmex.cpp ${OCATVE_LIBRARIES} ${MODELS} pbtlib matlabinterface)
        SET_TARGET_PROPERTIES(pbtoct PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/matlab")
endif (OCTAVE_FOUND)

