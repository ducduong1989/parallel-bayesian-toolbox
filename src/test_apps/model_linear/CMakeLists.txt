
##Define library name by folder name
SET (dir ${CMAKE_CURRENT_LIST_DIR}) 
string(REPLACE "/" ";" namelist ${dir})
LIST(LENGTH namelist len)
MATH( EXPR last "${len} - 1" )
LIST(GET namelist ${last} Name)

include_directories(${PROJECT_SOURCE_DIR}/src/models/model_linear)

ADD_EXECUTABLE( ${Name}Test test_linear.cpp)
target_link_libraries(${Name}Test pbtlib ${Name} ${ARMADILLO_LIBRARIES} hr_timer)

