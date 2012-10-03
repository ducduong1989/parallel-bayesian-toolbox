#ifndef MAPCONTROLPOINT_H
#define MAPCONTROLPOINT_H

#define MEDIA_TYPE_MAPCONTROLPOINT			0x00052001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_MAPCONTROLPOINT		0x00052001  

//#include <iostream>
#include <sstream>

struct MapControlPoint
{
	//absolute x (in meters)
	float x;
	//absolute y (in meters)
	float y; 
	//rotation (in radians)
	float maxSpeed; 

	friend	ostream &operator<<(ostream& os, const MapControlPoint& pt)
	{
		os << pt.x << ":" << pt.y << ":" << pt.maxSpeed;
		return os;
	};

	friend	istream &operator>>(istream& is, MapControlPoint& pt)
	{
		is >> pt.x;
		is >> pt.y;
		is >> pt.maxSpeed;

		return is;
	};
};

struct MapPoint
{
	//absolute x (in meters)
	float x;
	//absolute y (in meters)
	float y; 
};

#endif 
