#include "model_cuda.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

ModelCUDA::ModelCUDA(){

}

void ModelCUDA::setPNoise(){
}

void ModelCUDA::setONoise(){
}

fmat ModelCUDA::ffun(fmat *current){
    fmat prediction(0,0);

    return prediction;
}

fmat ModelCUDA::hfun(fmat *state){
    fmat measurement(0,0);

    return measurement;
}

void ModelCUDA::setDescriptionTag(){
    descriptionTag = "empty model";
}

frowvec ModelCUDA::eval(fmat* values){	//*< call of evaluation method for process noise batch
    frowvec evaluation;

    evaluation = pNoise.eval(values);

    return evaluation;
}
