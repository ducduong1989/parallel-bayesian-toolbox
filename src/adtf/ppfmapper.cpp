#include "ppfmapper.h"
#include "mapcontrolpoint.h"
#include <globalevents.h>

#ifndef M_PI
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#endif 


ADTF_FILTER_PLUGIN("Particle Filter based Mapper", OID_ADTF_PPFMapper, PPFMapper)

using namespace adtf;

PPFMapper::PPFMapper(const tChar *a_Info) :
    cFilter(a_Info)
{
    // REMARK: here you can setup your own properties!
    SetPropertyInt("PFilter:NumberOfParticles", 50);


    SetPropertyBool("FixedSpline", true);
    SetPropertyInt("NumberOfParticles",100);

    //GlobalReset();
}

tResult PPFMapper::Start(__exception /* = NULL */ )
{
    RETURN_NOERROR;
}


tResult PPFMapper::Init(cFilter::tInitStage eStage, __exception)
{
    RETURN_IF_FAILED(cFilter::Init(eStage, __exception_ptr));

    switch (eStage) {
    case StageFirst:
        RETURN_IF_FAILED(registerPins(__exception_ptr));
        break;

    case StageNormal:
        break;

    case StageGraphReady:
        m_fixedSpline = GetPropertyBool("FixedSpline") ;
        m_numberOfParticles = GetPropertyInt("NumberOfParticles");

        //include these Variables

        GlobalReset();

        memset(&m_outputData, 0, sizeof(m_outputData));

        THROW_IF_POINTER_NULL(_kernel);
        _kernel->SignalRegister(static_cast<IRunnable*>(this)); //register to receive kernel events

        //m_outputData.centerLineDistanceValid = false;
        //m_outputData.angleValid = false;
        break;
    }

    RETURN_NOERROR;
}

tResult PPFMapper::Shutdown(cFilter::tInitStage eStage, __exception)
{
    RETURN_IF_FAILED(cFilter::Shutdown(eStage, __exception_ptr));

    switch (eStage)
    {
    case StageGraphReady:
        THROW_IF_POINTER_NULL(_kernel);
        _kernel->SignalUnregister(static_cast<IRunnable*>(this)); //remove the registration for kernel events
        break;
    case StageNormal:
        break;
    case StageFirst:
        break;
    }

    RETURN_NOERROR;
}


tResult PPFMapper::OnPinEvent(adtf::IPin* pSource, tInt nEventCode, tInt nParam1, tInt nParam2, adtf::IMediaSample *pIMediaSample)
{
    if (nEventCode == IPinEventSink::PE_MediaSampleReceived) {
        if (pSource == &m_InSimplePose) {
            SimplePose *pInData = 0;
            RETURN_IF_FAILED(pIMediaSample->Lock((const tVoid **) &pInData));
            m_simplePose = *pInData;
            pIMediaSample->Unlock(pInData);
        }
        else if (pSource == &m_InVehicleState)
        {
            VehicleState *pInData = 0;
            RETURN_IF_FAILED(pIMediaSample->Lock((const tVoid **) &pInData));
            m_vehicleState = *pInData;
            pIMediaSample->Unlock(pInData);


            m_timeStamp = pIMediaSample->GetTime();

//            if(m_SimplePoseSample != NULL && m_VehicleStateSample != NULL/* && m_ControlPointsSample != NULL*/)
            {
                Process();
                cObjectPtr<adtf::IMediaSample> controllerOutput;
                RETURN_IF_FAILED(AllocMediaSample(&controllerOutput));
                RETURN_IF_FAILED(controllerOutput->Update(pIMediaSample->GetTime(), &m_outputData, sizeof(ControlData), 0));
            }
        }
        else if(pSource == &m_pinSpline)
        {
            SplinePoint *pInData = 0;
            RETURN_IF_FAILED(pIMediaSample->Lock((const tVoid **) &pInData));

            m_splinePoints.clear();
            int size = pIMediaSample->GetSize() / sizeof(SplinePoint);
            for (int i = 0; i < size; i++) {
                m_splinePoints.push_back(pInData[i]);
            }

            pIMediaSample->Unlock(pInData);
        }
    }

    RETURN_NOERROR;
}



tResult	PPFMapper::Process()
{
    std::vector<SplinePoint> splinePoints;

    if (m_fixedSpline) {
        splinePoints = m_splinePoints;
    } else {

        splinePoints.push_back(SplinePoint(1.398, 0.717));
        splinePoints.push_back(SplinePoint(1.105, 0.685));
        splinePoints.push_back(SplinePoint(0.870, 0.714));
        splinePoints.push_back(SplinePoint(0.709, 0.754));
        splinePoints.push_back(SplinePoint(0.550, 0.811));
        splinePoints.push_back(SplinePoint(0.419, 0.893));
        splinePoints.push_back(SplinePoint(0.267, 1.003));
        splinePoints.push_back(SplinePoint(0.141, 1.151));
        splinePoints.push_back(SplinePoint(-0.061, 1.287));
        splinePoints.push_back(SplinePoint(-0.295, 1.219));
        splinePoints.push_back(SplinePoint(-0.546, 1.093));
        splinePoints.push_back(SplinePoint(-0.701, 1.121));
        splinePoints.push_back(SplinePoint(-0.879, 1.132));
        splinePoints.push_back(SplinePoint(-1.042, 1.123));
        splinePoints.push_back(SplinePoint(-1.210, 1.086));
        splinePoints.push_back(SplinePoint(-1.356, 1.037));
        splinePoints.push_back(SplinePoint(-1.511, 0.936));
        splinePoints.push_back(SplinePoint(-1.775, 0.973));
        splinePoints.push_back(SplinePoint(-2.085, 0.968));
        splinePoints.push_back(SplinePoint(-2.218, 0.780));
        splinePoints.push_back(SplinePoint(-2.313, 0.584));
        splinePoints.push_back(SplinePoint(-2.396, 0.376));
        splinePoints.push_back(SplinePoint(-2.448, 0.140));
        splinePoints.push_back(SplinePoint(-2.415, -0.316));
        splinePoints.push_back(SplinePoint(-2.170, -0.938));
        splinePoints.push_back(SplinePoint(-1.694, -1.405));
        splinePoints.push_back(SplinePoint(-1.274, -1.595));
        splinePoints.push_back(SplinePoint(-0.666, -1.666));
        splinePoints.push_back(SplinePoint(-0.051, -1.656));
        splinePoints.push_back(SplinePoint(0.337, -1.668));
        splinePoints.push_back(SplinePoint(0.744, -1.454));
        splinePoints.push_back(SplinePoint(1.118, -1.244));
        splinePoints.push_back(SplinePoint(1.575, -1.212));
        splinePoints.push_back(SplinePoint(2.211, -1.238, 0));
        splinePoints.push_back(SplinePoint(2.537, -1.452, 0));
    }

    if (!splinePoints.empty())
        mapping(splinePoints, m_simplePose, m_vehicleState.speed, 0.5f);

    RETURN_NOERROR;
}

void PPFMapper::mapping(const std::vector<SplinePoint> & points, const SimplePose &pose, float speed, float steeringAngle)
{
    unsigned int mapDimX = 800;
    unsigned int mapDimY = 800;

    cv::Mat particles;
    cv::Mat processedMap;
    fmat samples;
    fmat map;
    fvec estimate;
    fvec odometryNoise = randn<fvec>(2)*0.1;
    int numberOfParticles = m_numberOfParticles;
    int dim = 3;
    fmat measurementRight = zeros<fmat>(35,2);

    float radiusCar = 150;
    float radiusRight = radiusCar + 17;
    for (unsigned int i = 0; i< measurementRight.n_rows; ++i){
        measurementRight(i,1)= -std::sqrt(radiusRight*radiusRight - i*i) + radiusCar;
        measurementRight(i,0)=i;
    }

    float radiusCenter = radiusRight - 35;
    fmat measurementCenter = zeros<fmat>(35,2);

    for (unsigned int i = 0; i< measurementCenter.n_rows; ++i){
        measurementCenter(i,1)=-std::sqrt(radiusCenter*radiusCenter - i*i) + radiusCar;
        measurementCenter(i,0)=i;
    }

    float radiusLeft = radiusCenter - 35;
    fmat measurementLeft = zeros<fmat>(10,2);

    for (unsigned int i = 0; i< measurementLeft.n_rows; ++i){
        measurementLeft(i,1)=-std::sqrt(radiusLeft*radiusLeft - i*i) + radiusCar;
        measurementLeft(i,0)=i;
    }

    /* initialize particle filter, estimator and resampler */
    filter = new BFilterSIR;
    resampler = new ResamplingMultinomial;
    estimator = new EstimationMean;
    /*EstimationKMeans *estimator = new EstimationKMeans;
    estimator->setConfiguration(2,25);
    estimator->setRefereceVector(zeros<fmat>(dim));*/
    //ModelCarLocalization *model = new ModelCarLocalization;
    model = new ModelCarSLAM;


    map = zeros<fmat>(mapDimY,mapDimX);
    model->setMapLeft(map);
    model->setMapCenter(map);
    model->setMapRight(map);

    // specify estimator and resampler that are used for particle filter
    filter->setResampling((Resampling*)resampler);
    filter->setEstimation((Estimation*)estimator);
    filter->setModel((Model*)model);

    //Start with an uniform particle distribution over the whole map
    samples = randu<fmat>(dim,numberOfParticles);
    //arrange them on startline
    for (unsigned int i=0;i<samples.n_cols;++i){
        samples(0,i) = 550;//+samples(0,i)*(rightLanes.cols-610);
        samples(1,i) = 300;// + samples(1,i)*(rightLanes.rows-610);
        //samples(2,i) = -0.001 - CV_PI/2 + samples(2,i)*0.002;
        samples(2,i) = - CV_PI/2;
    }

    filter->setParticles(samples,ones<frowvec>(numberOfParticles) * (1.0f/numberOfParticles));
    filter->setThresholdByFactor(0.5);

    map = model->getMap();
    timer.startTimer();
    for (unsigned int k = 0; k < 1000; ++k){
        // simulate a noisy odometry
        odometryNoise = randn<fvec>(2);
        model->speed(0)=4 + odometryNoise(0)*0.1;
        model->speed(1)=0;  // ;

        // stearing angle is not measured on car; just trust that servo sets the stearing angle correctly
        model->speed(2)= -0.021 + odometryNoise(1)*0.01;

        filter->predict();

        model->setRightSpline(measurementRight);
        model->setCenterSpline(measurementCenter);
        model->setLeftSpline(measurementLeft);
        filter->update(zeros<fvec>(measurementRight.n_rows
                                   +measurementCenter.n_rows
                                   +measurementLeft.n_rows));

        map = model->getMap();

        samples = filter->getParticles().samples;
        particles = cv::Mat::zeros(mapDimY,mapDimX, CV_8UC3);
        for (unsigned int i=0; i < samples.n_cols; ++i){
            int x = (int)samples(0,i);
            if ((x < 0) || (x >= (particles.cols-1))){
                x=0;
            }
            int y = (int)samples(1,i);
            if ((y < 0) || (y >= (particles.rows-1))){
                y=0;
            }
            cv::circle(particles,cv::Point(x,particles.rows-1-y),2,CV_RGB(255,0,0),2);
        }

        estimate = filter->getEstimation();
        //cv::circle(particles,cv::Point((int)estimate(0),particles.rows-(int)estimate(1)),5,CV_RGB(80,80,255),3);

        processedMap = processedMap + particles;
        imshow("Processed Map", processedMap);
        cvWaitKey(2);

    }
    timer.stopTimer();
    printf("Done! in %e seconds\n", timer.getElapsedTime());

    //cvDestroyWindow("orginal Image");
    cvDestroyWindow("Processed Map");

}


tResult PPFMapper::registerPins(__exception)
{
    cObjectPtr<adtf::IMediaType> pTypeSimplePose;
    RETURN_IF_FAILED(AllocMediaType((tVoid**)&pTypeSimplePose,
                                    MEDIA_TYPE_ROBOTPOSE,
                                    MEDIA_SUBTYPE_SIMPLEPOSE,
                                    NULL, NULL));

    cObjectPtr<adtf::IMediaType> pTypeVehicleState;
    RETURN_IF_FAILED(AllocMediaType((tVoid**)&pTypeVehicleState,
                                    MEDIA_TYPE_VEHICLESTATE,
                                    MEDIA_SUBTYPE_VS_STANDARD,
                                    NULL, NULL));

    cObjectPtr<adtf::IMediaType> pTypeControlData;
    RETURN_IF_FAILED(AllocMediaType((tVoid**)&pTypeControlData,
                                    MEDIA_TYPE_CONTROLDATA,
                                    MEDIA_SUBTYPE_CD_STANDARD,
                                    NULL, NULL));

    RETURN_IF_FAILED(m_InSimplePose.Create("Pose",pTypeSimplePose,this));
    RETURN_IF_FAILED(RegisterPin(&m_InSimplePose));

    RETURN_IF_FAILED(m_InVehicleState.Create("vehicle state", pTypeVehicleState, static_cast<IPinEventSink*>(this)));
    RETURN_IF_FAILED(RegisterPin(&m_InVehicleState));

    cObjectPtr<adtf::IMediaType> pType2 = NULL;
    RETURN_IF_FAILED(AllocMediaType(&pType2, MEDIA_TYPE_CONTROLLERINPUTDATA, MEDIA_SUBTYPE_CI_SPLINEPOINT, __exception_ptr));
    RETURN_IF_FAILED(m_pinSpline.Create("Spline", pType2, static_cast<IPinEventSink*>(this)));
    RETURN_IF_FAILED(RegisterPin(&m_pinSpline));

    RETURN_NOERROR;

}

tResult PPFMapper::Run(tInt nActivationCode,
                                   const tVoid* pvUserData,
                                   tInt szUserDataSize,
                                   ucom::IException** __exception_ptr/* =NULL */)
{
    tResult res = ERR_NOERROR;
    //this is the timer event
    if (nActivationCode == IRunnable::RUN_SIGNAL && pvUserData != NULL)
    {
        const adtf::tEventInfo* pEventInfo = (const adtf::tEventInfo*) pvUserData;
        if (pEventInfo->i32EventClass == adtf::IEvent::CLASS_CUSTOM &&
                pEventInfo->i32EventCode == FAUTO_EV_RESET)
        {
            LOG_INFO("PositionControllerMap: FAUTO_EV_RESET received - resetting!");
            GlobalReset();
        }
    }

    RETURN_IF_FAILED(res);
    return cFilter::Run(nActivationCode, pvUserData, szUserDataSize, __exception_ptr);
}

void PPFMapper::GlobalReset()
{
    //memset(&m_OutControlData, 0, sizeof(m_OutControlData));
    //memset(&m_debugOutputData, 0, sizeof(m_debugOutputData));
    //memset(&m_InSimplePose, 0, sizeof(m_InSimplePose));
    //memset(&m_InVehicleState, 0, sizeof(m_InVehicleState));

    m_inputDataList.clear();

    //initialize state
    m_inputSetPoint.angle = 0.0;
    m_inputSetPoint.speed = 0.0;
    m_inputSetPoint.centerLineDistance = 0.0;

}
