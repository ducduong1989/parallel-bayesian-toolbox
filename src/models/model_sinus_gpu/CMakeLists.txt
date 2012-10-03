
##Define library name by folder name
SET (dir ${CMAKE_CURRENT_LIST_DIR}) 
string(REPLACE "/" ";" namelist ${dir})
LIST(LENGTH namelist len)
MATH( EXPR last "${len} - 1" )
LIST(GET namelist ${last} Name)

# CUDA_ADD_LIBRARY(${Name} model_sinus_gpu.cpp model_sinus_gpu.h model_sinus_gpu.cu)
CUDA_ADD_LIBRARY(${Name} model_sinus_gpu.cpp model_sinus_gpu.h) # line above has to be inserted when cuda kernel file is in the repository
target_link_libraries(${Name} ppflib ${ARMADILLO_LIBRARIES})
