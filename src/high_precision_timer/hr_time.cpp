/* High Precision Timer found under:
* http://cplus.about.com/od/howtodothingsi2/a/timing.htm
* posted by David Bolton
*/

#include "hr_time.h"
#include <stdio.h>

#ifdef USE_WINDOWS_TIMER
#ifdef WINDOWS_32
#include <ctime>
#endif

#if WINDOWS_64
double CStopWatch::LIToSecs( LARGE_INTEGER & L) {
	//printf("q: %lu    h: %lu      l: %lu\n", L.QuadPart, L.HighPart, L.LowPart);
	return ((double)L.QuadPart /(double)frequency.QuadPart);


/*#if WINDOWS_32
	
	unsigned int lowpart = (unsigned int)frequency.LowPart;
	unsigned int highpart = (unsigned int)frequency.HighPart;
	unsigned int lowshift = 0;
	double pfreq = 0;

	while (highpart || (lowpart > 2000000.0))
	{
		lowshift++;
		lowpart >>= 1;
		lowpart |= (highpart & 1) << 31;
		highpart >>= 1;
	}

	pfreq = 1.0 / (double)lowpart;

	unsigned long long temp = ((unsigned int)L.LowPart >> lowshift) |
		((unsigned int)L.HighPart << (32 - lowshift));
	return 0;
#endif*/
}
#endif

CStopWatch::CStopWatch(){
#ifdef WINDOWS_32
	start = 0;
	stop = 0;
#endif
#if WINDOWS_64
	watch.start.QuadPart=0;
	watch.stop.QuadPart=0;	
	watch.start.HighPart=0;
	watch.start.LowPart=0;

	watch.stop.HighPart=0;
	watch.stop.LowPart=0;

	frequency.QuadPart=0;
	frequency.HighPart=0;
	frequency.LowPart=0;

	QueryPerformanceFrequency( &frequency );
#endif
}

void CStopWatch::startTimer( ) {
#ifdef WINDOWS_64
	QueryPerformanceCounter(&watch.start);
#endif
#ifdef WINDOWS_32
	start = 0;
	start = clock();
#endif
}

void CStopWatch::stopTimer( ) {
#ifdef WINDOWS_64
	QueryPerformanceCounter(&watch.stop);
#endif
#ifdef WINDOWS_32
	stop = 0;
	stop = clock();
#endif
}


double CStopWatch::getElapsedTime() {

#ifdef WINDOWS_64
	LARGE_INTEGER time;
	time.QuadPart = watch.stop.QuadPart - watch.start.QuadPart;
	time.HighPart = watch.stop.HighPart - watch.start.HighPart;
	time.LowPart = watch.stop.LowPart - watch.start.LowPart;
	
	return LIToSecs( time) ;
#endif
#ifdef WINDOWS_32
	//printf("32bit clock timer. Maybe uncertain.\n");
	return double(stop-start)/double(CLOCKS_PER_SEC);
#endif

}

CStopWatch::~CStopWatch(){
}

#endif

#ifdef USE_UNIX_TIMER

CStopWatch::CStopWatch(){
    start_sec = 0;
    start_usec = 0;
    stop_sec = 0;
    stop_usec = 0;
}

void CStopWatch::startTimer( ) {
	timeval start;
	gettimeofday(&start, 0);

	start_sec = start.tv_sec;
	start_usec = start.tv_usec;
}

void CStopWatch::stopTimer( ) {
	timeval stop;
	gettimeofday(&stop, 0);

	stop_sec = stop.tv_sec;
	stop_usec = stop.tv_usec;
}


double CStopWatch::getElapsedTime() {
	double elapsedTime = (stop_sec - start_sec);
	elapsedTime += (double)(stop_usec - start_usec) / 1000000.0;   // us to sec
	return elapsedTime;
}

CStopWatch::~CStopWatch(){

}

#endif

