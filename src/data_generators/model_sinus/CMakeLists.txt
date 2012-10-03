
##Define library name by folder name
SET (dir ${CMAKE_CURRENT_LIST_DIR}) 
string(REPLACE "/" ";" namelist ${dir})
LIST(LENGTH namelist len)
MATH( EXPR last "${len} - 1" )
LIST(GET namelist ${last} Name)

include_directories(${PROJECT_SOURCE_DIR}/src/models/model_sinus)

ADD_EXECUTABLE(sinusdatagen dataGenerator_sinus.cpp)
target_link_libraries(sinusdatagen ${Name} ppflib ${ARMADILLO_LIBRARIES})
