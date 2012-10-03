
include_directories(${PROJECT_SOURCE_DIR}/src/high_precision_timer)
include_directories(${PROJECT_SOURCE_DIR}/src/machine_info)
include_directories(${PROJECT_SOURCE_DIR}/src/models)


CUDA_ADD_EXECUTABLE(gputest gpu_test.cpp gpu_test.cu)

target_link_libraries(gputest cublas ${ARMADILLO_LIBRARIES}
                              hr_timer machineinfo model_wpam_3d_trajectory model_wpam_gpu)

