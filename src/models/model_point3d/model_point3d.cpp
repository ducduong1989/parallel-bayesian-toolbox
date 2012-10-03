#include "model_point3d.h"
#include "noise_gauss.h"
#include <math.h>

ModelPoint3D::ModelPoint3D(){

}

ModelPoint3D::~ModelPoint3D()
{

}

void ModelPoint3D::setDescriptionTag(){
    descriptionTag = "3d point";
}

void ModelPoint3D::setPNoise(){
    NoiseGaussian* x = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(y);
    NoiseGaussian* z = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(z);
}

void ModelPoint3D::setONoise(){
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(y);
    NoiseGaussian* z = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(z);
}

fmat ModelPoint3D::ffun(fmat *current){

    fmat pNoiseSample = pNoise.sample(current->n_cols);
    fmat prediction = *current + pNoiseSample;

    return prediction;
}

fmat ModelPoint3D::hfun(fmat *state){

    fmat measurement(state->n_rows,state->n_cols);
    fmat oNoiseSample = oNoise.sample(state->n_cols);

    measurement = *state + oNoiseSample;

    return measurement;
}

