/** \struct stopwatch High Resolution Timer structure found under:
  * http://cplus.about.com/od/howtodothingsi2/a/timing.htm
  * posted by David Bolton
  * \note modified to make the same structure usable on UNIX systems
  */

#ifndef __HR_TIMER__
#define __HR_TIMER__


#ifdef USE_UNIX_TIMER
    #include <sys/time.h>
#endif


/** \class CStopWatch High Resolution Timer class found under:
  * http://cplus.about.com/od/howtodothingsi2/a/timing.htm
  * posted by David Bolton
  * \note modified to make the same structure usable on UNIX systems
  */

#ifndef USE_WINDOWS_TIMER
class CStopWatch {

private:
    unsigned int start_sec;
    int start_usec;
    unsigned int stop_sec;
    int stop_usec;

public:
    CStopWatch();
    ~CStopWatch();
    void startTimer( );
    void stopTimer( );
    double getElapsedTime();
};
#endif

#ifdef USE_WINDOWS_TIMER

#include <windows.h>

#ifdef WINDOWS_64
typedef struct {
    LARGE_INTEGER start;
    LARGE_INTEGER stop;
} stopWatch;
#endif

class CStopWatch {

private:

#ifdef WINDOWS_64
    stopWatch watch;
    LARGE_INTEGER frequency;
    double LIToSecs( LARGE_INTEGER & L);
#endif
#ifdef WINDOWS_32
    unsigned int start;
    unsigned int stop;
#endif

public:
    CStopWatch();
    ~CStopWatch();
    void startTimer( );
    void stopTimer( );
    double getElapsedTime();
};

#endif

#endif
