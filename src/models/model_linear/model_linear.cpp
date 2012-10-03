#include "model_linear.h"
#include "noise_gauss.h"
#include <math.h>

void ModelLinear::setDescriptionTag(){
    descriptionTag = "Linear movement";
}

   void ModelLinear::setPNoise(){
        NoiseGaussian* x = new NoiseGaussian(0.0f, 0.01f);
        pNoise.addNoise(x);
        NoiseGaussian* y = new NoiseGaussian(0.0f, 0.01f);
        pNoise.addNoise(y);
    }

    void ModelLinear::setONoise(){
        NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.001f);
        oNoise.addNoise(x);
        NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.001f);
        oNoise.addNoise(y);
    }

    fmat ModelLinear::ffun(fmat *current){

        fmat prediction(current->n_rows,current->n_cols);
        fmat pNoiseSample = pNoise.sample(current->n_cols);

        prediction = *current + 0.5 + pNoiseSample;

        return prediction;
    }

    fmat ModelLinear::hfun(fmat *state){

        fmat measurement(state->n_rows,state->n_cols);
        fmat oNoiseSample = oNoise.sample(state->n_cols);

        measurement = *state + oNoiseSample;

        return measurement;
    }

