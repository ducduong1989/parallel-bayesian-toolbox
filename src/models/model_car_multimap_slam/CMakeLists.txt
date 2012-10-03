
##Define library name by folder name
SET (dir ${CMAKE_CURRENT_LIST_DIR}) 
string(REPLACE "/" ";" namelist ${dir})
LIST(LENGTH namelist len)
MATH( EXPR last "${len} - 1" )
LIST(GET namelist ${last} Name)

ADD_LIBRARY(${Name} model_car_multimap_slam.cpp model_car_multimap_slam.h)
target_link_libraries(${Name} ppflib ${ARMADILLO_LIBRARIES})

