include_directories(${PROJECT_SOURCE_DIR}/src/high_precision_timer)

#Test Applications
add_subdirectory (model_sinus)
add_subdirectory (model_linear)
add_subdirectory (model_fancy)
add_subdirectory (model_wpam_3d_trajectory)
add_subdirectory (model_straight3d)

if(OpenCV_FOUND)
    add_subdirectory (car_motion)
endif(OpenCV_FOUND)



