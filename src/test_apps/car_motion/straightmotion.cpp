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

#include "hr_time.h"

using namespace arma;

/**
  * Test application for car localization
  */
int main( int argc, char** argv )
{
    CStopWatch timer;
    cv::Mat originalImage = cv::imread("track_truth.png");
    cv::Mat blurImage = cv::imread("track_20px_blur.png");
    cv::Mat camImage = cv::Mat::zeros(50,50,CV_8UC3);
    int numMeasPoints=0;

    cv::VideoWriter record("Solution.avi", CV_FOURCC('D','I','V','X'), 30, originalImage.size(), true);
    if( !record.isOpened() )
    {
        printf("VideoWriter failed to open!\n");
        return -1;
    }

    cv::VideoCapture capture("simRobotVideo.avi");
    if( !capture.isOpened() )
    {
        printf("VideoPlayer failed to open!\n");
        return -1;
    }

    cv::Mat particles;
    fmat samples;
    fmat map;
    fvec estimate;
    //fvec odometryNoise = randn<fvec>(2)*0.1;
    int numberOfParticles = atoi(argv[1]);
    int dim = 3;
    fmat measurementRight = zeros<fmat>(35,2);

    fmat measurementCenter = zeros<fmat>(3,2);
    fmat measurementLeft = zeros<fmat>(3,2);


    /* initialize particle filter, estimator and resampler */
    BFilterSIR *filter = new BFilterSIR;
    ResamplingMultinomial *resampler = new ResamplingMultinomial;
    EstimationMean *estimator = new EstimationMean;
    /*EstimationKMeans *estimator = new EstimationKMeans;
    estimator->setConfiguration(2,25);
    estimator->setRefereceVector(zeros<fmat>(dim));*/
    //ModelCarLocalization *model = new ModelCarLocalization;
    ModelCarLocalization *model = new ModelCarLocalization;


    map = zeros<fmat>(blurImage.rows,blurImage.cols);
    model->setMapLeft(map);
    model->setMapCenter(map);

    for (unsigned int i = 0; i<blurImage.cols;++i){
        for (unsigned int j = 0; j< blurImage.rows; ++j){
            map(j,i) = blurImage.at<cv::Vec3b>(blurImage.rows-1-j,i)[0];
            map(j,i) += blurImage.at<cv::Vec3b>(blurImage.rows-1-j,i)[1];
            map(j,i) += blurImage.at<cv::Vec3b>(blurImage.rows-1-j,i)[2];
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
        samples(0,i) = samples(0,i)*(originalImage.cols);
        samples(1,i) = samples(1,i)*(originalImage.rows);
        //samples(2,i) = -0.001 - CV_PI/2 + samples(2,i)*0.002;
        samples(2,i) = 2*CV_PI*samples(2,i);
    }

    filter->setParticles(samples,ones<frowvec>(numberOfParticles) * (1.0f/numberOfParticles));
    filter->setThresholdByFactor(0.35);

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
        cv::circle(particles,cv::Point(x,particles.rows-1-y),2,CV_RGB(255,0,0),2);
    }
    imshow("Processed Map", originalImage + particles);
    record << (originalImage + particles);

    map = model->getMap();
    timer.startTimer();
    for (unsigned int k = 0; k < 95; ++k){
        capture >> camImage;
        cv::imshow("Window",camImage);
        cv::Mat resizedCam = cv::Mat::zeros(50,50,CV_8UC3);
        cv::resize(camImage,resizedCam,cv::Size(50,50));
        numMeasPoints=0;
        for (int i = 0; i< measurementRight.n_rows; ++i){
            for (int j = 0; j< resizedCam.cols; ++j){
                if (resizedCam.at<cv::Vec3b>(resizedCam.rows-1-i,j)[0] > 0){
                    measurementRight(i,1) = i;
                    measurementRight(i,0) = j - (resizedCam.cols/2);
                    numMeasPoints++;
                }
            }
        }


        model->speed(0) = 4;
        model->speed(1) = 0;
        model->speed(2) = 0;

        filter->predict();

        model->setRightSpline(measurementRight);
        model->setCenterSpline(measurementCenter);
        model->setLeftSpline(measurementLeft);
        filter->update(zeros<fvec>(measurementRight.n_rows
                                   +measurementCenter.n_rows
                                   +measurementLeft.n_rows));

        map = model->getMap();

        /*processedMap = cv::Mat::zeros(map.n_rows,map.n_cols, CV_8UC3);
        for (unsigned int i = 0; i<originalImage.cols;++i){
            for (unsigned int j = 0; j< originalImage.rows; ++j){
                if (map(j,i) > 0.0f){
                    processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[0] = 255*map(j,i);
                    processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[1] = 255*map(j,i);
                    processedMap.at<cv::Vec3b>(originalImage.rows-1-j,i)[2] = 255*map(j,i);
                }
            }
        }*/

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
            cv::circle(particles,cv::Point(x,particles.rows-1-y),2,CV_RGB(255,0,0),2);
        }

        estimate = filter->getEstimation();
        //cv::circle(particles,cv::Point((int)estimate(0),particles.rows-(int)estimate(1)),5,CV_RGB(80,80,255),3);

        imshow("Processed Map", originalImage + particles);
        record << (originalImage + particles);
        cvWaitKey(2);

    }
    timer.stopTimer();
    printf("Done! in %e seconds\n", timer.getElapsedTime());

    //cvDestroyWindow("orginal Image");
    cvDestroyWindow("Processed Map");

}
