## all implementations of noise sources in subdirectory
set(PPF_SUB_LIBRARIES)
add_subdirectory(noises)

## all library independent files for ppf library
set (PPFLIB_SOURCES
      ## helper functions
      ppf_helper_functions.cpp
      ## particle objects
      particle.cpp
      particles.cpp
      ## dummy model file
      model.cpp
      ## Particle Filter
      pf_sir.cpp
      pf_rpf.cpp
      pf_ekf.cpp
      pf_ukf.cpp
      pf_kld.cpp
      particlefilter.cpp
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
set (PPFLIB_HEADERS
      ## Header file to include all PBT classes
      pbt.h
      ## helper functions
      ppf_helper_functions.h
      ## particle objects
      particle.h 
      particles.h
      ## Particle Filter
      pf.h
      pf_sir.h
      pf_rpf.h
      pf_ekf.h
      pf_ukf.h
      pf_kld.h
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
    set (PPFLIB_SOURCES ${PPFLIB_SOURCES}
         estimation_mean_cuda.cpp
         estimation_robust_mean_cuda.cpp

         model_cuda.cpp

         pf_cuda.cpp
         pf_sir_cuda.cpp

         resampling_multinomial_cuda.cpp
    )
    set (PPFLIB_CUDA
         estimation_mean_cuda.cu
         estimation_robust_mean_cuda.cu

         helpful_cuda_kernels.cu

         pf_cuda.cu

         resampling_multinomial_cuda.cu
    )
    set (PPFLIB_HEADERS ${PPFLIB_HEADERS}
         estimation_mean_cuda.h
         estimation_robust_mean_cuda.h

         model_cuda.h

         pf_cuda.h
         pf_sir_cuda.h

         resampling_multinomial_cuda.h
    )
endif (CUDA_FOUND)

if (CUDA_FOUND)
    CUDA_ADD_LIBRARY(ppflib STATIC ${PPFLIB_SOURCES} ${PPFLIB_HEADERS} ${PPFLIB_CUDA} )
    target_link_libraries (ppflib ${ARMADILLO_LIBRARIES} noises)
else (CUDA_FOUND)
    ADD_LIBRARY(ppflib STATIC ${PPFLIB_SOURCES} ${PPFLIB_HEADERS} )
    target_link_libraries (ppflib ${ARMADILLO_LIBRARIES} noises)
endif (CUDA_FOUND)


if (CUDA_FOUND)    
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries (ppflib)
endif (CUDA_FOUND)
