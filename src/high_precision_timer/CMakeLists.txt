## High Precision Timer found under:
## http://cplus.about.com/od/howtodothingsi2/a/timing.htm
## posted by David Bolton

if(WIN32)
	add_definitions(-DUSE_WINDOWS_TIMER)

        IF(CMAKE_SIZEOF_VOID_P EQUAL 4)

            add_definitions(-DWINDOWS_32)

        ELSE(CMAKE_SIZEOF_VOID_P EQUAL 4)

            add_definitions(-DWINDOWS_64)

        ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 4)

else(WIN32)
        add_definitions(-DUSE_UNIX_TIMER)

endif(WIN32)


add_library(hr_timer hr_time.cpp hr_time.h)


