#include "estimation_mode.h"
#include <vector>
#include <iostream>

fmat EstimationMode::getEstimation(Particles* input){
    fvec  estimation;

    uvec sortedIndexdWeights = sort_index(input->weights,1);

    estimation = input->samples.col(sortedIndexdWeights(sortedIndexdWeights.n_rows));

    return estimation;
}

EstimationMode::~EstimationMode(){

}

EstimationMode::EstimationMode(){

}
