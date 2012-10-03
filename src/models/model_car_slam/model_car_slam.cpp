#include "model_car_slam.h"
#include "noise_gauss.h"
#include <cmath>
//#include <opencv2/opencv.hpp>

/*void ModelCarSLAM::showCam(fmat* cam)
{
    int greyscalingFactor = 255 * max(max(*cam));
    cv::Mat camView = cv::Mat::zeros(cam->n_rows,cam->n_cols, CV_8UC3);

    cvNamedWindow("CamView");

    for (unsigned int i = 0; i<camView.cols;++i){
        for (unsigned int j = 0; j< camView.rows; ++j){
            if (cam->at(j,i) > 0.01){
                camView.at<cv::Vec3b>(camView.rows-1-j,i)[0] = greyscalingFactor*cam->at(j,i);
                camView.at<cv::Vec3b>(camView.rows-1-j,i)[1] = greyscalingFactor*cam->at(j,i);
                camView.at<cv::Vec3b>(camView.rows-1-j,i)[2] = greyscalingFactor*cam->at(j,i);
            }
        }
    }
    imshow("Cam Image", camView);
    cvWaitKey(2);
}*/

void ModelCarSLAM::setDescriptionTag()
{
    descriptionTag = "Car modelwith SLAM";
}

void ModelCarSLAM::setPNoise()
{
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.1f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.1f);
    pNoise.addNoise(y);
    NoiseGaussian* phi = new NoiseGaussian( 0.0f, 0.03f);
    pNoise.addNoise(phi);
}

void ModelCarSLAM::setONoise()
{
    NoiseGaussian* vx = new NoiseGaussian( 0.0f, 0.1f);
    oNoise.addNoise(vx);

    NoiseGaussian* phiDev = new NoiseGaussian( 0.0f, 0.01f);
    oNoise.addNoise(phiDev);
}

fmat ModelCarSLAM::ffun(fmat *current)
{
    fmat prediction(current->n_rows,current->n_cols);
    fmat pNoiseSample = pNoise.sample(current->n_cols);

    fmat oNoiseSample = oNoise.sample(current->n_cols);

    // consider measurement noise
    /*for (unsigned int i=0; i< current->n_cols; ++i){
        prediction(0,i)= current->at(0,i) + deltaT*(speed(0)+oNoiseSample(0,i))*cos(current->at(2,i))  + deltaT*speed(1)*sin(current->at(2,i)) + pNoiseSample(0,i);
        prediction(1,i)= current->at(1,i) - deltaT*(speed(0)+oNoiseSample(0,i))*sin(current->at(2,i))  + deltaT*speed(1)*cos(current->at(2,i)) + pNoiseSample(1,i);
        prediction(2,i)= current->at(2,i) + deltaT*(speed(2)+oNoiseSample(1,i)) + pNoiseSample(2,i);
    }*/

    for (unsigned int i=0; i< current->n_cols; ++i){
        float cosPhi = cos(current->at(2,i));
        float sinPhi = sin(current->at(2,i));
        prediction(0,i)= current->at(0,i) + deltaT*speed(0)*cosPhi  + deltaT*speed(1)*sinPhi + pNoiseSample(0,i);
        prediction(1,i)= current->at(1,i) - deltaT*speed(0)*sinPhi  + deltaT*speed(1)*cosPhi + pNoiseSample(1,i);
        prediction(2,i)= current->at(2,i) + deltaT*speed(2) + pNoiseSample(2,i);
    }

    return prediction;
}

fmat ModelCarSLAM::hfun(fmat *state)
{
    fmat measurement;
    measurement = doLocalizationMeasurements(state);

    // map of right road marking
    // generate view of camera from spline
    fmat noisyCamView = generateCamView(&splineRight);
    // apply gaussian noise to the camera view to avoid a one pixel wide line
    //fmat noisyCamView = applyGaussianNoise(&camView);
    //noisyCamView = applyGaussianNoise(&noisyCamView);
    // update map parameters
    updateHitsAndMisses(state, &noisyCamView, &hitsRight, &missesRight);

    // map of center road marking
    // generate view of camera from spline
    noisyCamView = generateCamView(&splineCenter);
    // apply gaussian noise to the camera view to avoid a one pixel wide line
    //noisyCamView = applyGaussianNoise(&camView);
    //noisyCamView = applyGaussianNoise(&noisyCamView);
    // update map parameters
    updateHitsAndMisses(state, &noisyCamView, &hitsCenter, &missesCenter);

    // map of left road marking
    // generate view of camera from spline
    noisyCamView = generateCamView(&splineLeft);
    // apply gaussian noise to the camera view to avoid a one pixel wide line
    //noisyCamView = applyGaussianNoise(&camView);
    //noisyCamView = applyGaussianNoise(&noisyCamView);
    // update map parameters
    updateHitsAndMisses(state, &noisyCamView, &hitsLeft, &missesLeft);

    return measurement;
}



void ModelCarSLAM::updateHitsAndMisses(fmat* state, fmat* camView, fmat* hits, fmat* misses)
{
    //showCam(camView);
    for (unsigned int i = 0; i<state->n_cols; ++i){
        float stateX = state->at(0,i);
        float stateY = state->at(1,i);
        float statePhi = state->at(2,i) ;

        float sinPhi = std::sin(statePhi);
        float cosPhi = std::cos(statePhi);

        for (unsigned int posX=0; posX< viewX; posX++)
        {
            float cosPosX = cosPhi * posX;
            float sinPosX = sinPhi * posX;
            for (unsigned int posY=0; posY< viewY; posY++)
            {
                int mapPositionX = (int)((stateX + cosPosX + sinPhi*(posY-float(zeroY)))/spacingX);
                int mapPositionY = (int)((stateY - sinPosX + cosPhi*(posY-float(zeroY)))/spacingY);

                if ((mapPositionX >= 0)&&(mapPositionX < (int)hits->n_cols))
                {
                    if ((mapPositionY >= 0)&&(mapPositionY < (int)hits->n_rows))
                    {
                        if (camView->at(posY,posX) == 0)
                        {
                            misses->at(mapPositionY, mapPositionX) += 1;
                        }
                        else
                        {
                            // fill hits and misses  relating to the position in map
                            hits->at(mapPositionY,mapPositionX) += 4;// camView->at(posY,posX);
                        }
                    }
                    else printf("Cam out of map in Y: %i", mapPositionY);
                }
                else printf("Cam out of map in X: %i", mapPositionX);
            }
        }
    }
}

fmat ModelCarSLAM::applyGaussianNoise(fmat* input)
{
    fmat output = zeros<fmat>(input->n_rows,input->n_cols);
    for (unsigned int posX=1; posX< input->n_cols-1; posX++)
    {
        for (unsigned int posY=1; posY< input->n_rows-1; posY++)
        {
            output(posY,posX) += 6*input->at(posY,posX);
            output(posY,posX) += input->at(posY-1,posX);
            output(posY,posX) += input->at(posY+1,posX);
            output(posY,posX) += input->at(posY,posX-1);
            output(posY,posX) += input->at(posY,posX+1);

            output(posY,posX) = output(posY,posX) / 10.0f;
        }
    }

    output = output / max(max(output));

    return output;
}

fmat ModelCarSLAM::generateCamView(fmat* spline)
{
    fmat camView = zeros<fmat>(viewY,viewX);
    for (unsigned int j=0; j < spline->n_rows; ++j)
    {
        unsigned int camPositionX = zeroX + int(spline->at(j,0)/spacingX);
        if (camPositionX<0){
            camPositionX = 0;
            printf("Overflow!\n");
        }
        else {
            if (camPositionX >= camView.n_cols){
                camPositionX = camView.n_cols-1;
                printf("Overflow!\n");
            }
        }

        unsigned int camPositionY = zeroY + int(spline->at(j,1)/spacingY);
        if (camPositionY<0){
            camPositionY = 0;
            printf("Overflow!\n");
        }
        else {
            if (camPositionY >= camView.n_rows){
                camPositionY = camView.n_rows-1;
                printf("Overflow!\n");
            }
        }

        camView(camPositionY, camPositionX) += 1;
    }
    return camView;
}

fmat ModelCarSLAM::doLocalizationMeasurements(fmat *state)
{
    int numberOfMesurements = splineRight.n_rows + splineCenter.n_rows + splineLeft.n_rows;
    int measurementOffset = 0;
    fmat measurement = zeros<fmat>(numberOfMesurements,state->n_cols);

    for (unsigned int i=0; i< measurement.n_cols; ++i){
        measurementOffset = 0;
        float cosPhi = std::cos(state->at(2,i));
        float sinPhi = std::sin(state->at(2,i));

        for (unsigned int j=0; j < splineRight.n_rows; ++j){
            int mapPositionX = int((state->at(0,i) + cosPhi*splineRight(j,0) + sinPhi*splineRight(j,1))/spacingX);
            if (mapPositionX<0){
                mapPositionX = 0;
            }
            else {
                if (mapPositionX >= (int)hitsRight.n_cols){
                    mapPositionX = hitsRight.n_cols-1;
                }
            }

            int mapPositionY = int((state->at(1,i) - sinPhi*splineRight(j,0) + cosPhi*splineRight(j,1))/spacingY);
            if (mapPositionY<0){
                mapPositionY = 0;
            }
            else {
                if (mapPositionY >= (int)hitsRight.n_rows){
                    mapPositionY = hitsRight.n_rows-1;
                }
            }

            measurement(j,i) = hitsRight(mapPositionY, mapPositionX) /
                               (hitsRight(mapPositionY, mapPositionX) +  missesRight(mapPositionY, mapPositionX));
        }

        measurementOffset += splineRight.n_rows;

        for (unsigned int j=0; j < splineCenter.n_rows; ++j){
            int mapPositionX = int((state->at(0,i) + cosPhi*splineCenter(j,0) + sinPhi*splineCenter(j,1))/spacingX);
            if (mapPositionX<0){
                mapPositionX = 0;
            }
            else {
                if (mapPositionX >= (int)hitsCenter.n_cols){
                    mapPositionX = hitsCenter.n_cols-1;
                }
            }

            int mapPositionY = int((state->at(1,i) - sinPhi*splineCenter(j,0) + cosPhi*splineCenter(j,1))/spacingY);
            if (mapPositionY<0){
                mapPositionY = 0;
            }
            else {
                if (mapPositionY >= (int)hitsCenter.n_rows){
                    mapPositionY = hitsCenter.n_rows-1;
                }
            }

            measurement(measurementOffset+j,i) = hitsCenter(mapPositionY, mapPositionX) /
                                                (hitsCenter(mapPositionY, mapPositionX) +  missesCenter(mapPositionY, mapPositionX));
        }

        measurementOffset += splineCenter.n_rows;

        for (unsigned int j=0; j < splineLeft.n_rows; ++j){
            int mapPositionX = int((state->at(0,i) + cosPhi*splineLeft(j,0) + sinPhi*splineLeft(j,1))/spacingX);
            if (mapPositionX<0){
                mapPositionX = 0;
            }
            else {
                if (mapPositionX >= (int)hitsLeft.n_cols){
                    mapPositionX = hitsLeft.n_cols-1;
                }
            }

            int mapPositionY = int((state->at(1,i) - sinPhi*splineLeft(j,0) + cosPhi*splineLeft(j,1))/spacingY);
            if (mapPositionY<0){
                mapPositionY = 0;
            }
            else {
                if (mapPositionY >= (int)hitsLeft.n_rows){
                    mapPositionY = hitsLeft.n_rows-1;
                }
            }

            measurement(measurementOffset+j,i) = hitsLeft(mapPositionY, mapPositionX) /
                                                (hitsLeft(mapPositionY, mapPositionX) +  missesLeft(mapPositionY, mapPositionX));
        }
    }

    return measurement;
}


frowvec ModelCarSLAM::eval(fmat *values){	//*< call of evaluation method for process noise batch
    frowvec evaluation = zeros<frowvec>(values->n_cols);
    for (unsigned int i=0; i<values->n_cols;++i){
        for (unsigned int j=0; j<values->n_rows;++j){
            //debug message
            if (values->at(j,i) <0 ){
                perror("\nmap point smaller than zero: %e\n");
                printf("\nmap point smaller than zero: %e\n", values->at(j,i));
            }
            evaluation(i) += values->at(j,i);
        }
    }
    return evaluation;
}


ModelCarSLAM::ModelCarSLAM(){
    viewX = 40;
    viewY = 80;

    zeroX = 0;
    zeroY = 25;

    mapCellsX = 5;
    mapCellsY = 5;

    spacingX = 1.0f;
    spacingY = 1.0f;

    hitsRight = ones<fmat>(mapCellsY,mapCellsX);
    missesRight = ones<fmat>(mapCellsY,mapCellsX);

    hitsCenter = ones<fmat>(mapCellsY,mapCellsX);
    missesCenter = ones<fmat>(mapCellsY,mapCellsX);

    hitsLeft = ones<fmat>(mapCellsY,mapCellsX);
    missesLeft = ones<fmat>(mapCellsY,mapCellsX);

    deltaT = 1;

    speed = zeros<fvec>(3);
}

fmat ModelCarSLAM::getMap(){
    fmat map;
    mapRight = hitsRight / (hitsRight+missesRight);
    mapCenter = hitsCenter / (hitsCenter+missesCenter);
    mapLeft = hitsLeft / (hitsLeft+missesLeft);

    map = mapRight + mapCenter + mapLeft;
    //map = map / max(max(map));

    return map;
}

void ModelCarSLAM::setMapRight(fmat newMap){
    mapRight = newMap;
    //hits = map*1000;
    //misses = (1-map)*1000;

    hitsRight = ones<fmat>(newMap.n_rows,newMap.n_cols);
    missesRight = ones<fmat>(newMap.n_rows,newMap.n_cols);
}

void ModelCarSLAM::setMapCenter(fmat newMap){
    mapCenter = newMap;
    //hits = map*1000;
    //misses = (1-map)*1000;

    hitsCenter = ones<fmat>(newMap.n_rows,newMap.n_cols);
    missesCenter = ones<fmat>(newMap.n_rows,newMap.n_cols);
}

void ModelCarSLAM::setMapLeft(fmat newMap){
    mapLeft = newMap;
    //hits = map*1000;
    //misses = (1-map)*1000;

    hitsLeft = ones<fmat>(newMap.n_rows,newMap.n_cols);
    missesLeft = ones<fmat>(newMap.n_rows,newMap.n_cols);
}

void ModelCarSLAM::setRightSpline(fmat splinepoints){
    splineRight = splinepoints;
}

void ModelCarSLAM::setCenterSpline(fmat splinepoints){
    splineCenter = splinepoints;
}

void ModelCarSLAM::setLeftSpline(fmat splinepoints){
    splineLeft = splinepoints;
}

ModelCarSLAM::~ModelCarSLAM(){

}
