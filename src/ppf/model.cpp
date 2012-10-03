#include "model.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

Model::Model(){

}

void Model::setPNoise(){
}

void Model::setONoise(){
}

fmat Model::ffun(fmat *current){
    fmat prediction(0,0);

    return prediction;
}

fmat Model::hfun(fmat *state){
    fmat measurement(0,0);

    return measurement;
}

void Model::setDescriptionTag(){
    descriptionTag = "empty model";
}

frowvec Model::eval(fmat* values){	//*< call of evaluation method for process noise batch
    frowvec evaluation;

    evaluation = pNoise.eval(values);

    return evaluation;
}

unsigned int Model::getProcessDimension(){
    return pNoise.sample(0).n_rows;
}

unsigned int Model::getMeasurementDimension(){
    return oNoise.sample(0).n_rows;
}

void Model::initialize(){
    setPNoise();
    setONoise();
    setDescriptionTag();

    setF();
    setH();
}

NoiseBatch Model::getProcessNoise(){
    return pNoise;
}

NoiseBatch Model::getObservationNoise(){
    return oNoise;
}

void Model::setProcessNoise(NoiseBatch newPNoise){
    pNoise = newPNoise;
}

void Model::setObservationNoise(NoiseBatch newONoise){
    oNoise = newONoise;
}

void Model::setF(){
    H = eye<fmat>(1,1);
}

void Model::setH(){
    H = eye<fmat>(1,1);
}
