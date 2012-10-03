Find_Package(OpenCV)

if(OpenCV_FOUND)


include_directories(${PROJECT_SOURCE_DIR}/src/models/model_car_localization)

include_directories(OpenCV_INCLUDE_DIRS)
set (OpenCV_LIBRARIES ${OpenCV_LIBRARIES} ${OpenCV_LIBS})

add_executable(motion_extractor motion_extractor.cpp)

target_link_libraries(motion_extractor model_car_localization ${OpenCV_LIBRARIES} ${ARMADILLO_LIBRARIES})

endif(OpenCV_FOUND)
