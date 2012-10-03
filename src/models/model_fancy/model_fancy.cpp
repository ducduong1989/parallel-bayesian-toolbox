#include "model_fancy.h"
#include "noise_gauss.h"
#include <math.h>
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14
#endif

void ModelFancy::setDescriptionTag(){
    descriptionTag = "wave package model";
}

    void ModelFancy::setPNoise(){
        NoiseGaussian* x = new NoiseGaussian(0.0f, 0.0001f);
        pNoise.addNoise(x);
        NoiseGaussian* y = new NoiseGaussian(0.0f, 0.0001f);
        pNoise.addNoise(y);
    }

    void ModelFancy::setONoise(){
        NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.001f);
        oNoise.addNoise(x);
        NoiseGaussian* y = new NoiseGaussian( 1.0f, 0.01f);
        oNoise.addNoise(y);
    }

    fmat ModelFancy::ffun(fmat *current){

        fmat prediction(current->n_rows,current->n_cols);
        fmat pNoiseSample = pNoise.sample(current->n_cols);
        float x = 0;

        for (unsigned int i=0;i<current->n_cols;++i){
            prediction(0,i) = current->at(0,i)+0.05f;
            x = prediction(0,i);

            prediction(1,i) = (float) sin(x*3.0f)*10.0f*(1.0f/(float)(3.0f*sqrt(2.0f*M_PI)))*exp(-0.5f*pow((x-11.0f)/3.0f,2.0f));
        }

        prediction = prediction + pNoiseSample;

        return prediction;
    }

    fmat ModelFancy::hfun(fmat *state){

        fmat measurement(state->n_rows,state->n_cols);
        fmat oNoiseSample = oNoise.sample(state->n_cols);

        measurement = *state + oNoiseSample;

        return measurement;
    }
