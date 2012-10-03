
find_package(ADTF
  QUIET
  PATHS
  /opt/adtf
  /home/*/adtf
  /proj/i4ezs/tools/ADTF
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Elektrobit Automotive GmbH\\ADTF\\2;InstallDir]"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Wow6432Node\\Elektrobit Automotive GmbH\\ADTF\\2;InstallDir]"
)

message (STATUS "ADTF found: ${ADTF_FOUND}")

if (ADTF_FOUND)
	

        set(PPF_ADTF_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin" CACHE PATH "ADTF output path")

        adtf_output_dir(${PPF_ADTF_OUTPUT_PATH})

        include_directories(${ARMADILLO_INCLUDE_DIR})
        include_directories(include)
        include_directories(${PROJECT_SOURCE_DIR}/src/high_precision_timer)
        include_directories(${PROJECT_SOURCE_DIR}/src/models/model_car_localization)
        include_directories(${PROJECT_SOURCE_DIR}/src/models/model_car_slam)

        set(HEADERS
            include/calibrationstate.h
            include/CMakeLists.txt
            include/controldata.h
            include/controllerinput.h
            include/fullposeeuler.h
            include/globalevents.h
            include/laserdata.h
            include/mapcontrolpoint.h
            include/obstacledata.h
            include/pathplannerstate.h
            include/simplepose.h
            include/stoplinedata.h
            include/vehiclestate.h
        )

	set(SOURCES
	    ppfmapper.cpp
 	    ppfmapper.h
	)

        adtf_add_filter(ppfmapper ${SOURCES} ${HEADERS})

        target_link_libraries(ppfmapper ${OpenCV_LIBRARIES} ${OpenCV_LIBS} ppflib model_car_localization model_car_slam hr_timer)

endif (ADTF_FOUND)
