include_directories(OpenCV_INCLUDE_DIRS)
include_directories(${PROJECT_SOURCE_DIR}/src/models/model_car_localization)
include_directories(${PROJECT_SOURCE_DIR}/src/models/model_car_slam)


add_executable(carmotion carmotion.cpp)
set (OpenCV_LIBRARIES ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(carmotion ${OpenCV_LIBRARIES} pbtlib model_car_localization model_car_slam)
#target_link_libraries(carmotion ${OpenCV_LIBRARIES} pbtlib model_car_slam)


add_executable(circlemotion circlemotion.cpp)

target_link_libraries(circlemotion ${OpenCV_LIBRARIES} pbtlib model_car_localization model_car_slam hr_timer)

add_executable(straightmotion straightmotion.cpp)

target_link_libraries(straightmotion ${OpenCV_LIBRARIES} pbtlib model_car_localization model_car_slam hr_timer)

