#include "model_straight3d.h"
#include "noise_gauss.h"
#include <math.h>

ModelStraight3D::ModelStraight3D(){
    directionVector = ones<fvec>(3);
    pointOfOrigin = zeros<fvec>(3);

    H = eye<fmat>(3,3);

    H = eye<fmat>(3,3);
}

ModelStraight3D::~ModelStraight3D()
{

}

void ModelStraight3D::setDescriptionTag(){
    descriptionTag = "3d straight";
}

void ModelStraight3D::setPNoise(){
    NoiseGaussian* x = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(y);
    NoiseGaussian* z = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(z);
}

void ModelStraight3D::setONoise(){
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(y);
    NoiseGaussian* z = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(z);
}

fmat ModelStraight3D::ffun(fmat *current){

    fmat prediction(current->n_rows,current->n_cols);
    fmat pNoiseSample = pNoise.sample(current->n_cols);

    // simple assumption
    for (unsigned int i=0; i< current->n_cols; ++i)
    {
        prediction.col(i) = current->col(i) + directionVector + pNoiseSample.col(i);
    }

    // more complex approach
    /*for (unsigned int i = 0; i< current->n_cols; ++i)
    {
        // find nearest point on defined straight

        // add vector

        // add noise

    }*/

    return prediction;
}

fmat ModelStraight3D::hfun(fmat *state){

    fmat measurement(state->n_rows,state->n_cols);
    fmat oNoiseSample = oNoise.sample(state->n_cols);

    measurement = *state + oNoiseSample;

    return measurement;
}


void ModelStraight3D::addStraightDefinition(fvec origin, fvec vector)
{
    pointOfOrigin = origin;
    directionVector = vector;
}

