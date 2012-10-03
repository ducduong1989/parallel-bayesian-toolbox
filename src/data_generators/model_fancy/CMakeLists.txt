
##Define library name by folder name
SET (dir ${CMAKE_CURRENT_LIST_DIR}) 
string(REPLACE "/" ";" namelist ${dir})
LIST(LENGTH namelist len)
MATH( EXPR last "${len} - 1" )
LIST(GET namelist ${last} Name)

include_directories(${PROJECT_SOURCE_DIR}/src/models/model_fancy)

ADD_EXECUTABLE(fancydatagen dataGenerator_fancy.cpp)
target_link_libraries(fancydatagen ${Name} ppflib ${ARMADILLO_LIBRARIES})
