#include "model_car_localization.h"
#include "noise_gauss.h"
#include <math.h>

void ModelCarLocalization::setDescriptionTag(){
    descriptionTag = "Car motion with localization";
}

void ModelCarLocalization::setPNoise(){
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.0f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.0f);
    pNoise.addNoise(y);
    NoiseGaussian* phi = new NoiseGaussian( 0.0f, 0.0f);
    pNoise.addNoise(phi);
}

void ModelCarLocalization::setONoise(){
    NoiseGaussian* all;
    for (unsigned int i=0; i<10;++i){
        all = new NoiseGaussian(0.0f, 1.0f);
        oNoise.addNoise(all);
    }
}

fmat ModelCarLocalization::ffun(fmat *current){

    fmat prediction(current->n_rows,current->n_cols);
    fmat pNoiseSample = pNoise.sample(current->n_cols);

    for (unsigned int i=0; i< current->n_cols; ++i){
        prediction(0,i)= current->at(0,i) + deltaT*speed(0)*cos(current->at(2,i))  + deltaT*speed(1)*sin(current->at(2,i)) + pNoiseSample(0,i);
        prediction(1,i)= current->at(1,i) - deltaT*speed(0)*sin(current->at(2,i))  + deltaT*speed(0)*cos(current->at(2,i)) + pNoiseSample(1,i);
        prediction(2,i)= current->at(2,i) + deltaT*speed(2) + pNoiseSample(2,i);
    }

    return prediction;
}

fmat ModelCarLocalization::hfun(fmat *state){
    int numberOfMesurements = splineRight.n_rows + splineCenter.n_rows + splineLeft.n_rows;
    int measurementOffset = 0;
    fmat measurement = zeros<fmat>(numberOfMesurements,state->n_cols);

    for (unsigned int i=0; i< measurement.n_cols; ++i){
        for (unsigned int j=0; j < splineRight.n_rows; ++j){
            int mapPositionX = int((state->at(0,i) + cos(state->at(2,i))*splineRight(j,0) + sin(state->at(2,i))*splineRight(j,1))/spacingX);
            if (mapPositionX<0){
                mapPositionX = 0;
            }
            else {
                if (mapPositionX >= (int)mapRight.n_cols){
                    mapPositionX = mapRight.n_cols-1;
                }
            }
            float sinPhi = sinf(state->at(2,i));
            float cosPhi = cosf(state->at(2,i));
            float splineX = splineRight(j,0);
            float splineY = splineRight(j,1);

            int mapPositionY = int((state->at(1,i) - sinPhi*splineX + cosPhi*splineY)/spacingY);
            if (mapPositionY<0){
                mapPositionY = 0;
            }
            else {
                if (mapPositionY >= (int)mapRight.n_rows){
                    mapPositionY = mapRight.n_rows-1;
                }
            }

            measurement(j,i) = mapRight(mapPositionY, mapPositionX);
        }
    }
    measurementOffset += splineRight.n_rows;

    for (unsigned int i=0; i< measurement.n_cols; ++i){
        for (unsigned int j=0; j < splineCenter.n_rows; ++j){
            int mapPositionX = int((state->at(0,i) + cos(state->at(2,i))*splineCenter(j,0) + sin(state->at(2,i))*splineCenter(j,1))/spacingX);
            if (mapPositionX<0){
                mapPositionX = 0;
            }
            else {
                if (mapPositionX >= (int)mapCenter.n_cols){
                    mapPositionX = mapCenter.n_cols-1;
                }
            }

            int mapPositionY = int((state->at(1,i) - sin(state->at(2,i))*splineCenter(j,0) + cos(state->at(2,i))*splineCenter(j,1))/spacingY);
            if (mapPositionY<0){
                mapPositionY = 0;
            }
            else {
                if (mapPositionY >= (int)mapCenter.n_rows){
                    mapPositionY = mapCenter.n_rows-1;
                }
            }

            measurement(measurementOffset+j,i) = mapCenter(mapPositionY, mapPositionX);
        }
    }

    measurementOffset += splineCenter.n_rows;

    for (unsigned int i=0; i< measurement.n_cols; ++i){
        for (unsigned int j=0; j < splineLeft.n_rows; ++j){
            int mapPositionX = int((state->at(0,i) + cos(state->at(2,i))*splineLeft(j,0) + sin(state->at(2,i))*splineLeft(j,1))/spacingX);
            if (mapPositionX<0){
                mapPositionX = 0;
            }
            else {
                if (mapPositionX >= (int)mapLeft.n_cols){
                    mapPositionX = mapLeft.n_cols-1;
                }
            }

            int mapPositionY = int((state->at(1,i) - sin(state->at(2,i))*splineLeft(j,0) + cos(state->at(2,i))*splineLeft(j,1))/spacingY);
            if (mapPositionY<0){
                mapPositionY = 0;
            }
            else {
                if (mapPositionY >= (int)mapLeft.n_rows){
                    mapPositionY = mapLeft.n_rows-1;
                }
            }

            measurement(measurementOffset+j,i) = mapLeft(mapPositionY, mapPositionX);
        }
    }
    //measurement.print();
    return measurement;
}

frowvec ModelCarLocalization::eval(fmat* values){	//*< call of evaluation method for process noise batch
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
    //evaluation.print();
    return evaluation;
}


ModelCarLocalization::ModelCarLocalization(){
    mapCellsX = 5;
    mapCellsY = 5;

    zeroCellX = 2;
    zeroCellY = 2;

    spacingX = 1;
    spacingY = 1;

    //hits = zeros<fmat>(mapCellsY,mapCellsX);
    //misses = zeros<fmat>(mapCellsY,mapCellsX);

    deltaT = 1;

    speed = zeros<fvec>(3);
}

fmat ModelCarLocalization::getMap(){
    //mapRight = hits / (hits+misses);
    fmat map = mapRight + mapCenter + mapCenter;
    return map;
}

void ModelCarLocalization::setMapRight(fmat newMap){
    mapRight = newMap;
}

void ModelCarLocalization::setMapCenter(fmat newMap){
    mapCenter = newMap;
}

void ModelCarLocalization::setMapLeft(fmat newMap){
    mapLeft = newMap;
}

void ModelCarLocalization::setRightSpline(fmat splinepoints){
    splineRight = splinepoints;
}

void ModelCarLocalization::setCenterSpline(fmat splinepoints){
    splineCenter = splinepoints;
}

void ModelCarLocalization::setLeftSpline(fmat splinepoints){
    splineLeft = splinepoints;
}

ModelCarLocalization::~ModelCarLocalization(){

}
