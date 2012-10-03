#ifndef LASERDATA_H
#define LASERDATA_H

#define MEDIA_TYPE_LASERDATA		0x00080001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_XYFLOAT		0x00081001  
#define MEDIA_SUBTYPE_XYDOUBLE		0x00081002
#define MEDIA_SUBTYPE_XYINT			0x00081003
#define MEDIA_SUBTYPE_DISTANCERAW	0x00081004

#define MEDIA_TYPE_OBJDESC			0x00080010  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_DETECT		0x00081011  
#define MEDIA_SUBTYPE_TRACKED		0x00081012

struct LaserDataRaw
{
	//laser measurement starting angle in radians
	float startAngle;
	//laser measurement step angle in radians
	float stepAngle;
	//scale to convert raw mesaures to meters (e.g. 1.0f/1000.0f - for distances sent in milimiters)
	float scaleToMeters; 
	//number of measurements sent in the packet
	unsigned int count;		
	//here folows the raw distance measures (d0, d1, d2, ...) in unsigned int words
};

template <class T>
struct LaserPoint
{
	T x;
	T y;
};


struct	ObjDescription
{
	// object position clockwise
	float	p0;
	float	p1;
	float	p2;
	float	p3;
	
	// object velocity
	float	vx;
	float	vy;	
};

#endif // LASERDATA_H
