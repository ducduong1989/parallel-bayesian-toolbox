#ifndef PPFMAPPER_H
#define PPFMAPPER_H

//#include <cv.h>

#include <armadillo>
#include <opencv2/opencv.hpp>

#include <adtf_plugin_sdk.h>
#include "controldata.h"
#include "controllerinput.h"
#include "mapcontrolpoint.h"
#include <iostream>
#include <list>
#include "vehiclestate.h"
#include "simplepose.h"

//#include "model_car_localization.h"
#include "model_car_slam.h"
#include "pf_sir.h"
#include "resampling_multinomial.h"
//#include "resampling_multinomial_cuda.h"
#include "estimation_mean.h"
#include "estimation_kmeans.h"
#include "hr_time.h"
#include <fstream>
#include <adtf_plugin_sdk.h>


#define OID_ADTF_PPFMapper "fautonom.ppf_mapper"

class PPFMapper : public adtf::cFilter
{
    ADTF_FILTER(OID_ADTF_PPFMapper, "Particle Filter based Mapper", adtf::OBJCAT_DataFilter);

public:
    PPFMapper(const tChar *a_Info);

	void GlobalReset();


public:
    tResult Init(cFilter::tInitStage eStage, __exception);
	tResult Shutdown(cFilter::tInitStage eStage, __exception);
    tResult OnPinEvent(adtf::IPin *pSource, tInt nEventCode, tInt nParam1, tInt nParam2, adtf::IMediaSample *pIMediaSample);
	tResult	Start(__exception /* = NULL */);
	//tResult Stop();

	tResult Run(tInt nActivationCode, const tVoid* pvUserData, tInt szUserDataSize, __exception);

private:

    tResult		registerPins(__exception);
    tResult		Process();

    void mapping(const std::vector<SplinePoint> & points, const SimplePose &pose, float speed, float steeringAngle);

    adtf::cInputPin			m_InSimplePose;
    adtf::cInputPin			m_InVehicleState;
    adtf::cInputPin			m_pinSpline;
	//
    adtf::cOutputPin		m_OutSimplePose;
    adtf::cVideoPin         m_OutMap;

    LaneTrackingData			m_inputData;
    ControllerInputSetPoint		m_inputSetPoint;
    std::list<LaneTrackingData> m_inputDataList;
    ControlData					m_outputData;
    VehicleState m_vehicleState;
    SimplePose m_simplePose;

    //include these Variables
    int   m_numberOfParticles;

    CStopWatch timer;

    BFilterSIR *filter;
    ResamplingMultinomial *resampler;
    EstimationMean *estimator;

    ModelCarSLAM *model;

    tTimeStamp m_timeStamp;

    std::vector<SplinePoint> m_splinePoints;
    bool m_fixedSpline;
};


#endif // PPFMAPPER_H
