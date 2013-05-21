## Noise sources defined in GUM

set(NOISE_HEADERS   pbtnoises.h
                    noise_gauss.h
                    noise_gamma.h
                    noise_exp.h
                    noise_trap.h
                    noise_ctrap.h
                    noise_tri.h
                    noise_u.h
                    noise_uni.h
                    noise_studentt.h)

set(NOISE_SOURCES   noise_gauss.cpp
                    noise_gamma.cpp
                    noise_exp.cpp
                    noise_trap.cpp
                    noise_ctrap.cpp
                    noise_tri.cpp
                    noise_u.cpp
                    noise_uni.cpp
                    noise_studentt.cpp)

if(CUDA_FOUND)
    set(NOISE_HEADERS   ${NOISE_HEADERS}
                        noise_gauss_gpu.h)

    set(NOISE_SOURCES   ${NOISE_SOURCES}
                        noise_gauss_gpu.cpp)

    set(NOISE_CUDA_KERNELS   noise_gauss_gpu.cu)

    CUDA_ADD_LIBRARY(noises ${NOISE_HEADERS} ${NOISE_SOURCES} ${NOISE_CUDA_KERNELS})

else(CUDA_FOUND)

    add_library(noises ${NOISE_HEADERS} ${NOISE_SOURCES})

endif(CUDA_FOUND)

target_link_libraries(noises pbtlib)

set(PBT_SUB_LIBRARIES ${PBT_SUB_LIBRARIES} noises)
