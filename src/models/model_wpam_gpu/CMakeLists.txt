
##Define library name by folder name
SET (dir ${CMAKE_CURRENT_LIST_DIR}) 
string(REPLACE "/" ";" namelist ${dir})
LIST(LENGTH namelist len)
MATH( EXPR last "${len} - 1" )
LIST(GET namelist ${last} Name)

CUDA_ADD_LIBRARY(${Name} model_wpam_gpu.cpp model_wpam_gpu.h model_wpam_gpu.cu)
target_link_libraries(${Name} ppflib curand ${ARMADILLO_LIBRARIES})
