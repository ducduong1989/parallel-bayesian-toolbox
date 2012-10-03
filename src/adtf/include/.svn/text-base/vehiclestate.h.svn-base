#ifndef VEHICLESTATE_H
#define VEHICLESTATE_H


//#include <stdint.h>

#define MEDIA_TYPE_VEHICLESTATE			0x00060001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_VS_STANDARD		0x00061001  
#define MEDIA_SUBTYPE_VS_GOFLAG			0x00061002

struct	GoFlag
{
	bool	goFlag;
};

struct VehicleState
{
    enum CarMode { CarIdle, CarDebug };

    // todo change naming
    //uint32_t timeStamp;
    unsigned int timeStamp;
    CarMode carMode;
    bool rcOverride;
    bool pcReceived;
    float speed; // m/s
    float steering; // rad
    float encoderDistance; // m
    float distanceFront; // m
    float distanceRightFront; // m
    float distanceRightBack; // m
    float distanceBackLeft; // m
    float distanceBackRight; // m

    bool button1;
    bool button2;
    bool button3;
    /*
	unsigned int time;
	float throttle;
	float target_speed;
	float current_speed;
	float current_distance;
	float target_distance;
        float steering;

        unsigned short sharp_front;
        unsigned short sharp_right_front;
        unsigned short sharp_right_back;
        unsigned short sharp_back_left;
        unsigned short sharp_back_right;

	unsigned int bytes_received;
	unsigned short packets_received;
	unsigned char packet_errors;
	char mode;
	char rc;
        */
};


#endif // VEHICLESTATE_H
