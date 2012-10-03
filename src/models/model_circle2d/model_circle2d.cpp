#include "model_circle2d.h"
#include "noise_gauss.h"
#include <math.h>

ModelCircle2D::ModelCircle2D(){

}

ModelCircle2D::~ModelCircle2D()
{

}

void ModelCircle2D::setDescriptionTag(){
    descriptionTag = "2d circle";
}

void ModelCircle2D::setRadius(float circleRadius)
{
    radius = circleRadius;
}

void ModelCircle2D::setPhiStep(float phi)
{
    phiStep = phi;
}

void ModelCircle2D::setOrigin(fvec origin)
{
    orig = origin;
}

void ModelCircle2D::setPNoise(){
    NoiseGaussian* x = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian(0.0f, 0.01f);
    pNoise.addNoise(y);
}

void ModelCircle2D::setONoise(){
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.06f);
    oNoise.addNoise(y);
}

fmat ModelCircle2D::ffun(fmat *current){

    fmat pNoiseSample = pNoise.sample(current->n_cols);

    fmat prediction = pNoiseSample;
    float phi = 0;

    for (unsigned int i = 0; i < current->n_cols; ++i)
    {
        phi = std::atan2(current->at(1,i), current->at(0,i));
        phi += phiStep;
        prediction(0,i) += radius * std::cos(phi);
        prediction(1,i) += radius * std::sin(phi);
    }

    return prediction;
}

fmat ModelCircle2D::hfun(fmat *state){

    fmat measurement(state->n_rows,state->n_cols);
    fmat oNoiseSample = oNoise.sample(state->n_cols);

    measurement = *state + oNoiseSample;

    return measurement;
}

