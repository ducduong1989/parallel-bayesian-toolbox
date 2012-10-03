#ifndef CONTROLLERINPUT_H
#define CONTROLLERINPUT_H

#define MEDIA_TYPE_CONTROLLERINPUTDATA	0x000A0001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_CI_LANETRACKING	0x000A1001
#define MEDIA_SUBTYPE_CI_SETPOINT		0x000A1002
#define MEDIA_SUBTYPE_CI_SPLINEPOINT		0x000A1003
#define MEDIA_SUBTYPE_CI_LANETRACKINGCLOTHOID	0x000A1004

struct LaneTrackingData
{
    double speed; // m/s
    double centerLineDistance; // m
    double angle; // rad

    double quality; // quality of the tracking: 0..1.0, 0 worst, 1.0 best

    double timeStamp; // us
};

struct LaneTrackingDataClothoid
{
	double centerLineDistance; // m
	double angle; // rad
	double curvature;
	double curvatureChange;
	double width; //in CaroloCup, fixed to 0.42m

	double quality; // quality of the tracking: 0..1.0, 0 worst, 1.0 best
};

struct ControllerInputSetPoint
{
    double speed; // m/s
    double centerLineDistance; // m
    double angle;  // rad

    bool indicatorLeft;
    bool indicatorRight;
    bool brake;
    bool light;
    bool gimmick;

    bool debugGreen;
    bool debugYellow;
    bool debugRed;
    bool debugMix;
};

struct SplinePoint
{
    SplinePoint() {}
    SplinePoint(float x, float y, float speed = 100) : x(x), y(y), speed(speed) {}
    float x;
    float y;
    float speed;
};

#endif // CONTROLLERINPUT_H

