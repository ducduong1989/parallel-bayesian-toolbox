#include <opencv2/opencv.hpp>
#include "model_car_localization.h"
#include "model_car_slam.h"
#include "pf_sir.h"
#include "resampling_multinomial.h"
#include "resampling_multinomial_cuda.h"
#include "estimation_mean.h"
#include "estimation_kmeans.h"
#include <fstream>
#include <armadillo>

using namespace arma;

/**
  * Test application for car localization
  */
int main( int argc, char** argv )
{
    cv::Mat originalImage = cv::imread("track_truth.png");

    cv::Mat rightLanes = cv::imread("track_right_blur.png");

    cv::Mat particles;
    cv::Mat processedMap;
    fmat samples;
    fmat map;
    fvec estimate;
    fvec odometryNoise = randn<fvec>(2)*0.1;
    int numberOfParticles = atoi(argv[1]);
    int dim = 3;
    fmat measurementRight = zeros<fmat>(35,2);

    for (int i = 0; i< measurementRight.n_rows; ++i){
        measurementRight(i,1)=-17;
        measurementRight(i,0)=i;
    }

    fmat measurementCenter = zeros<fmat>(35,2);

    for (int i = 0; i< measurementCenter.n_rows; ++i){
        measurementCenter(i,1)=+17;
        measurementCenter(i,0)=i;
    }

    fmat measurementLeft = zeros<fmat>(35,2);

    for (int i = 0; i< measurementLeft.n_rows; ++i){
        measurementLeft(i,1)=+51;
        measurementLeft(i,0)=i;
    }

    /* initialize particle filter, estimator and resampler */
    BFilterSIR *filter = new BFilterSIR;
    ResamplingMultinomial *resampler = new ResamplingMultinomial;
    EstimationMean *estimator = new EstimationMean;
    /*EstimationKMeans *estimator = new EstimationKMeans;
    estimator->setConfiguration(2,25);
    estimator->setRefereceVector(zeros<fmat>(dim));*/
    //ModelCarLocalization *model = new ModelCarLocalization;
    ModelCarSLAM *model = new ModelCarSLAM;


    map = zeros<fmat>(rightLanes.rows,rightLanes.cols);
    model->setMapLeft(map);
    model->setMapCenter(map);

    for (unsigned int i = 0; i<rightLanes.cols;++i){
        for (unsigned int j = 0; j< rightLanes.rows; ++j){
            map(j,i) = rightLanes.at<cv::Vec3b>(rightLanes.rows-1-j,i)[0];
            map(j,i) += rightLanes.at<cv::Vec3b>(rightLanes.rows-1-j,i)[1];
            map(j,i) += rightLanes.at<cv::Vec3b>(rightLanes.rows-1-j,i)[2];
            if (map(j,i)<0){
                map(j,i) = 0;
            }
        }
    }

    model->setMapRight(map);

    // specify estimator and resampler that are used for particle filter
    filter->setResampling((Resampling*)resampler);
    filter->setEstimation((Estimation*)estimator);
    filter->setModel((Model*)model);

    //Start with an uniform particle distribution over the whole map
    samples = randu<fmat>(dim,numberOfParticles);
    //arrange them on startline
    for (unsigned int i=0;i<samples.n_cols;++i){
        samples(0,i) = 570;//+samples(0,i)*(rightLanes.cols-610);
        samples(1,i) = 120;// + samples(1,i)*(rightLanes.rows-610);
        //samples(2,i) = -0.001 - CV_PI/2 + samples(2,i)*0.002;
        samples(2,i) = - CV_PI/2;
    }

    filter->setParticles(samples,ones<frowvec>(numberOfParticles) * (1.0f/numberOfParticles));
    filter->setThresholdByFactor(0.5);

    map = model->getMap();

    processedMap = cv::Mat::zeros(map.n_rows,map.n_cols, CV_8UC3);
    for (unsigned int i = 0; i<originalImage.cols;++i){
        for (unsigned int j = 0; j< originalImage.rows; ++j){
            if (map(j,i) > 0.01){
                processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[0] = 200*map(j,i);
                processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[1] = 200*map(j,i);
                processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[2] = 200*map(j,i);
            }
        }
    }

    //cvNamedWindow("Orginal Image");
    cvNamedWindow("Processed Map");

    for (unsigned int k = 0; k < 500; ++k){
        // simulate a noisy odometry
        odometryNoise = randn<fvec>(2);
        model->speed(0)=4 + odometryNoise(0)*0.1;
        model->speed(1)=0;  // ;

        // stearing angle is not measured on car; just trust that servo sets the stearing angle correctly
        model->speed(2)= 0;// + odometryNoise(1)*0.03;

        filter->predict();

        imshow("Processed Map", processedMap + particles);
        cvWaitKey(2);

        model->setRightSpline(measurementRight);
        model->setCenterSpline(measurementCenter);
        model->setLeftSpline(measurementLeft);
        filter->update(zeros<fvec>(measurementRight.n_rows
                                   +measurementCenter.n_rows
                                   +measurementLeft.n_rows));

        map = model->getMap();

        processedMap = cv::Mat::zeros(map.n_rows,map.n_cols, CV_8UC3);
        for (unsigned int i = 0; i<originalImage.cols;++i){
            for (unsigned int j = 0; j< originalImage.rows; ++j){
                if (map(j,i) > 0.0f){
                    processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[0] = 255*map(j,i);
                    processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[1] = 255*map(j,i);
                    processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[2] = 255*map(j,i);
                }
            }
        }

        samples = filter->getParticles().samples;
        particles = cv::Mat::zeros(originalImage.rows,originalImage.cols, CV_8UC3);
        for (unsigned int i=0; i < samples.n_cols; ++i){
            int x = (int)samples(0,i);
            if ((x < 0) || (x >= (particles.cols-1))){
                x=0;
            }
            int y = (int)samples(1,i);
            if ((y < 0) || (y >= (particles.rows-1))){
                y=0;
            }
            cv::circle(particles,cv::Point(x,particles.rows-1-y),1,CV_RGB(255,0,0),1);
        }

        estimate = filter->getEstimation();
        cv::circle(particles,cv::Point((int)estimate(0),particles.rows-(int)estimate(1)),5,CV_RGB(80,80,255),3);

        imshow("Processed Map", processedMap + particles);
        cvWaitKey(2);

    }

    //cvDestroyWindow("orginal Image");
    cvDestroyWindow("Processed Map");

}
