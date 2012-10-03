#ifndef CONTROLDATA_H
#define CONTROLDATA_H

#include <adtf_plugin_sdk.h>

#define MEDIA_TYPE_CONTROLDATA			0x00070001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_CD_STANDARD		0x00071001
#define MEDIA_SUBTYPE_CD_DEBUG			0x00071002

struct ControlData
{
    tFloat64 speed; // m/s
    tFloat64 steering;  // rad

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

struct ControlDebugData
{
    tFloat64 timeDiff;
    tFloat64 cldDiff;
    tFloat64 angleDiff;

    tFloat64 errorCld;
    tFloat64 errorAngle;

    // for debugging purpose
    tFloat64 steeringAnglePValue;  // rad
    tFloat64 steeringCldPValue;    // rad
    tFloat64 steeringAngleDValue;  // rad
    tFloat64 steeringCldDValue;    // rad
};

#endif // CONTROLDATA_H
