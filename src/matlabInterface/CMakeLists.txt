SET(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${PROJECT_SOURCE_DIR}/src/ppf)
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
        MATLAB_MEX_CREATE(ppfmex ppfmex.cpp ${MATLAB_LIBRARIES} ${MODELS} ppflib matlabinterface)
        SET_TARGET_PROPERTIES(ppfmex PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/matlab")
endif (MATLAB_FOUND)

if (OCTAVE_FOUND)
        OCTAVE_MEX_CREATE(ppfoct ppfmex.cpp ${OCATVE_LIBRARIES} ${MODELS} ppflib matlabinterface)
        SET_TARGET_PROPERTIES(ppfoct PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/matlab")
endif (OCTAVE_FOUND)

