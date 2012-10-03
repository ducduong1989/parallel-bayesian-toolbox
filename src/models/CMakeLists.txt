
add_subdirectory (model_sinus)
add_subdirectory (model_linear)
add_subdirectory (model_point3d)
add_subdirectory (model_circle2d)
add_subdirectory (model_straight3d)
add_subdirectory (model_fancy)
add_subdirectory (model_wpam_3d_trajectory)
add_subdirectory (model_car_localization)
add_subdirectory (model_car_slam)
add_subdirectory (model_car_multimap_slam)

if (CUDA_FOUND)
    add_subdirectory (model_sinus_gpu)
    add_subdirectory (model_wpam_gpu)
endif(CUDA_FOUND)




