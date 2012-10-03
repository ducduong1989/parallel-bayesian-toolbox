#ifndef ROBOTPOSESIMPLE_H
#define ROBOTPOSESIMPLE_H

#define MEDIA_TYPE_ROBOTPOSE			0x00042001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_SIMPLEPOSE		0x00042001  
#define MEDIA_SUBTYPE_POSEDIFFERENCE	0x00042002  

class SimplePose
{
public:
	//absolute x (in meters)
	float x;
	//absolute y (in meters)
	float y; 
	//rotation (in radians)
	float angle; 
	
	SimplePose(float _x = 0.0f, float _y = 0.0f, float _angle = 0.0f) {x = _x; y = _y; angle = _angle;};
        
	SimplePose operator-(const SimplePose& other) const
	{
		SimplePose difference;
	
		difference.x = x - other.x;
		difference.y = y - other.y;
		difference.angle = angle - other.angle;
	
		return difference;
	}
	SimplePose operator+(const SimplePose& other) const
	{
		SimplePose difference;

		difference.x = x + other.x;
		difference.y = y + other.y;
		difference.angle = angle + other.angle;

		return difference;
	}  
	
	bool operator==(const SimplePose & other) const
	{
		if ((x == other.x)&&(y == other.y)&&(angle == other.angle)) return true;
	
		return false;
	}        
};

#endif //ROBOTPOSESIMPLE_H
