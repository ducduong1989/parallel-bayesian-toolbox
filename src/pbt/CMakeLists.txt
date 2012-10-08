## all implementations of noise sources in subdirectory
set(PBT_SUB_LIBRARIES)
add_subdirectory(noises)

## all library independent files for ppf library
set (PBTLIB_SOURCES
      ## helper functions
      ppf_helper_functions.cpp
      ## particle objects
      particle.cpp
      particles.cpp
      ## dummy model file
      model.cpp
      ## Bayes Filter
      bf.cpp
      bf_sirpf.cpp
      bf_rpf.cpp
      bf_ekf.cpp
      bf_ukf.cpp
      bf_kld.cpp
      bf_llpf.cpp
      ## Noise Batch
      noise_batch.cpp
      ## Noise sources
      noise.cpp
      ## Estimation ##
      estimation.cpp 
      estimation_mean.cpp 
      estimation_kmeans.cpp
      estimation_mean_shift.cpp
      estimation_median.cpp
      estimation_mode.cpp
      estimation_modema.cpp 
      estimation_best_particle.cpp
      estimation_robust_mean.cpp
      ## Resampling ##
      resampling.cpp
      resampling_multinomial.cpp 
      resampling_residual.cpp 
      resampling_systematic.cpp
      resampling_stratified.cpp
      resampling_naive_delete.cpp
)
set (PBTLIB_HEADERS
      ## Header file to include all PBT classes
      pbt.h
      ## helper functions
      ppf_helper_functions.h
      ## particle objects
      particle.h 
      particles.h
      ## Bayes Filter
      bf.h
      bf_sirpf.h
      bf_rpf.h
      bf_ekf.h
      bf_ukf.h
      bf_kld.h
      bf_llpf.h
      ## Model definition
      model.h 
      ## Noise Batch
      noise_batch.h
      ## Noise source
      noise.h
      ## Estimation ##
      estimation.h
      estimation_mean.h 
      estimation_kmeans.h
      estimation_mean_shift.h
      estimation_median.h
      estimation_mode.h
      estimation_modema.h
      estimation_best_particle.h
      estimation_robust_mean.h
      ## Resampling ##
      resampling.h
      resampling_multinomial.h 
      resampling_residual.h 
      resampling_systematic.h 
      resampling_stratified.h 
      resampling_naive_delete.h
)

#if CUDA is on system implementations with CUDA support are added
if (CUDA_FOUND)
    set (PBTLIB_SOURCES ${PBTLIB_SOURCES}
         estimation_mean_cuda.cpp
         estimation_robust_mean_cuda.cpp

         model_cuda.cpp

         bf_cuda.cpp
         bf_sirpf_cuda.cpp

         resampling_multinomial_cuda.cpp
    )
    set (PBTLIB_CUDA
         estimation_mean_cuda.cu
         estimation_robust_mean_cuda.cu

         helpful_cuda_kernels.cu

         bf_cuda.cu

         resampling_multinomial_cuda.cu
    )
    set (PBTLIB_HEADERS ${PBTLIB_HEADERS}
         estimation_mean_cuda.h
         estimation_robust_mean_cuda.h

         model_cuda.h

         bf_cuda.h
         bf_sirpf_cuda.h

         resampling_multinomial_cuda.h
    )
endif (CUDA_FOUND)

if (CUDA_FOUND)
    CUDA_ADD_LIBRARY(pbtlib STATIC ${PBTLIB_SOURCES} ${PBTLIB_HEADERS} ${PBTLIB_CUDA} )
    target_link_libraries (pbtlib ${ARMADILLO_LIBRARIES} noises)
else (CUDA_FOUND)
    ADD_LIBRARY(pbtlib STATIC ${PBTLIB_SOURCES} ${PBTLIB_HEADERS} )
    target_link_libraries (pbtlib ${ARMADILLO_LIBRARIES} noises)
endif (CUDA_FOUND)


if (CUDA_FOUND)    
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries (pbtlib)
endif (CUDA_FOUND)
