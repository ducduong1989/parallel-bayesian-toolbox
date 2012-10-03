#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <armadillo>
#include "model_car_localization.h"

using namespace arma;

int initStateFile(char *file)
{
    std::ofstream out(file,std::ios_base::trunc);

    if (!out)
    {
        perror("output file error\n");
        return -1;
    }

    out.close();

    return 0;
}


void addState(char *file, fmat u)
{
    std::ofstream out(file,std::ios_base::app);
    for(unsigned int i=0; i < u.n_rows; ++i){
        out << u[i] << " ";
    }
    out << std::endl;
    out.close();
}

void extractWindow(cv::Mat* map, cv::Mat* window, float posX, float posY, float phi ,int sizeX, int sizeY){
    float x;
    float y;
    float tempPhi = CV_PI-0.05;
    for (int i = 0; i < sizeY; ++i){
        for (int j=0; j< sizeX;++j){
            x = j - (sizeX/2);
            y = i;
            int xSinPhi = x*sinf(tempPhi);
            int xCosPhi = x*cosf(tempPhi);
            int yCosPhi = y*cosf(tempPhi);
            int ySinPhi = y*sinf(tempPhi);
            window->at<cv::Vec3b>(i,j)[0] = map->at<cv::Vec3b>(posY-xSinPhi+yCosPhi, posX+xCosPhi-ySinPhi)[0];
            window->at<cv::Vec3b>(i,j)[1] = map->at<cv::Vec3b>(posY-xSinPhi+yCosPhi, posX+xCosPhi-ySinPhi)[1];
            window->at<cv::Vec3b>(i,j)[2] = map->at<cv::Vec3b>(posY-xSinPhi+yCosPhi, posX+xCosPhi-ySinPhi)[2];
        }
    }

    //return window;

}


std::vector<float> extractRightLinePositions(cv::Mat* view){
    std::vector<float> spline(view->rows);
    uchar maxval = 0;
    uchar maxPosX = 0;
    bool foundPointInLine = false;

    for (int i = 0; i < view->rows; ++i){
        maxval = 0;
        foundPointInLine = false;
        for (int j=0; j<view->cols;++j){
            if ( maxval < view->at<uchar>(i, j)) {
                foundPointInLine = true;
                maxval = view->at<uchar>(i, j);
                maxPosX = j;
            }

        }
        if (foundPointInLine){
            spline.push_back(maxPosX);
            spline.push_back(i);
        }
    }

    return spline;
}


int main( int argc, char** argv )
{
    /*if (argc<2){
        perror("Error");
        printf("Please call program like:\n");
        printf("./datagen filename_original filename_measurement numberOfStates\n");
        return -1;
    }*/

    int windowWidth = 60;
    int windowHeight = 60;
    int num = 50;

    //char *file_original = argv[1];
    //char *file_out = argv[2];
    num = atoi(argv[1]);

    char* fileAddress_truth=(char*)"track_truth.png";
    cv::Mat map = cv::imread(fileAddress_truth);
    cv::Mat correctMap = cv::Mat::zeros(map.rows,map.cols,CV_8UC3);

    for (int i=0; i< map.rows; i++){
        for(int j=0; j< map.cols; j++){
            correctMap.at<cv::Vec3b>(i,j)[0] = map.at<cv::Vec3b>(map.rows-1-i, j)[0];
            correctMap.at<cv::Vec3b>(i,j)[1] = map.at<cv::Vec3b>(map.rows-1-i, j)[1];
            correctMap.at<cv::Vec3b>(i,j)[2] = map.at<cv::Vec3b>(map.rows-1-i, j)[2];
        }
    }

    cv::Mat particles = cv::Mat::zeros(correctMap.rows,correctMap.cols, CV_8UC3);


    cv::Mat window = cv::Mat::zeros(windowHeight,windowWidth, CV_8UC3);

    cv::Mat lanes = cv::imread("track_truth.png");

    cv::VideoWriter record("simRobotVideo.avi", CV_FOURCC('D','I','V','X'), 30, cv::Size(500,500), true);
    if( !record.isOpened() )
    {
        printf("VideoWriter failed to open!\n");
        return -1;
    }

    cv::VideoWriter positions("posRobotVideo.avi", CV_FOURCC('D','I','V','X'), 30, lanes.size(), true);


    //std::vector<float> spline;

    //float startX = 10;
    //float startY = 10;
    //float startPhi = 0;



    ModelCarLocalization *process = new ModelCarLocalization();
    process->initialize();

    printf("Dimension of state: %d\n",process->getProcessDimension());

    fmat state(process->getProcessDimension(),1);

    //fmat measurement(process->getMeasurementDimension(),1);

    state.zeros();
    state(0) = 428;
    state(1) = 605;
    state(2) = CV_PI-0.05;

    //initStateFile(file_out);	// initialize output file
    //initStateFile(file_original);

    //addState(file_out, state);

    process->speed(0) = 4;
    process->speed(1) = 0;
    process->speed(2) = 0;

    for (int k=0; k<num;k++)
    {
        // generate a new position
        state = process->ffun(&state);
        printf("x: %e  y: %e\n",state(0),state(1));
        //addState(file_original,state);

        float x;
        float y;
        float tempPhi = CV_PI+0.75;
        for (int i = 0; i < windowHeight; ++i){
            for (int j=0; j< windowWidth;++j){
                x = j - (windowWidth/2);
                y = i;
                int xSinPhi = x*sinf(tempPhi);
                int xCosPhi = x*cosf(tempPhi);
                int yCosPhi = y*cosf(tempPhi);
                int ySinPhi = y*sinf(tempPhi);
                window.at<cv::Vec3b>(windowHeight-1-i,j)[0] = correctMap.at<cv::Vec3b>(state(1)-xSinPhi+yCosPhi, state(0)+xCosPhi+ySinPhi)[0];
                window.at<cv::Vec3b>(windowHeight-1-i,j)[1] = correctMap.at<cv::Vec3b>(state(1)-xSinPhi+yCosPhi, state(0)+xCosPhi+ySinPhi)[1];
                window.at<cv::Vec3b>(windowHeight-1-i,j)[2] = correctMap.at<cv::Vec3b>(state(1)-xSinPhi+yCosPhi, state(0)+xCosPhi+ySinPhi)[2];
            }
        }

        imshow("Window", window);
        cvWaitKey(2);

        //extractWindow(&correctMap, &window, float(state(0)), float(state(1)), float(state(2)), windowWidth, windowHeight );

        //spline = extractRightLinePositions(&window);

        //measurement = zeros<fmat>(spline.size());

        //for (unsigned int i = 0; i < spline.size(); ++i){
        //    measurement(i) = spline.at(i);
        //}

        // extract points from map
        //measurement = process->hfun(&state);
        //printf("x: %e  y: %e\n\n",measurement(0),measurement(1));
        //addState(file_out, measurement);

        cv::circle(particles,cv::Point(int(state(0)),particles.rows-1-(state(1))),2,CV_RGB(255,0,0),2);
        positions << (map+particles);
        record << window;

    }

    printf("Done!");

    return 0;

}

