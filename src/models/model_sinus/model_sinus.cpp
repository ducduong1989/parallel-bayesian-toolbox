#include "model_sinus.h"
#include <math.h>
#include "noise_gauss.h"

void ModelSinus::setDescriptionTag(){
    descriptionTag = "Sinus signal model";
}

void ModelSinus::setPNoise(){
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.05f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.05f);
    pNoise.addNoise(y);
}

void ModelSinus::setONoise(){
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.1f);
    oNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.1f);
    oNoise.addNoise(y);
}

fmat ModelSinus::ffun(fmat *current){

    fmat prediction(current->n_rows,current->n_cols);
    fmat pNoiseSample = pNoise.sample(current->n_cols);

    for (unsigned int i=0;i<current->n_cols;++i){
        prediction(0,i) = current->at(0,i)+0.1f;

        prediction(1,i) = sin(prediction(0,i));
    }

    prediction = prediction + pNoiseSample;

    return prediction;
}

fmat ModelSinus::hfun(fmat *state){

    fmat measurement(state->n_rows,state->n_cols);
    fmat oNoiseSample = oNoise.sample(state->n_cols);

    measurement = *state + oNoiseSample;

    return measurement;
}
