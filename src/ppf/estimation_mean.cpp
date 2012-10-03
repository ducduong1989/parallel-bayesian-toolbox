#include "estimation_mean.h"
#include <vector>
#include <iostream>

fmat EstimationMean::getEstimation(Particles* input){
    float sum=0.0;
    fvec  estimation(input->samples.n_rows);
#ifdef VERBOSE
        printf("calculating estimation of %d particles in %d dimensions\n",input->samples.n_cols,input->samples.n_rows);
#endif

    for (unsigned int j=0;j<input->samples.n_rows;++j){
            sum = 0.0;
            for (unsigned int i=0; i<input->samples.n_cols; ++i){
                sum += input->samples(j,i) * input->weights(i);
            }
            estimation(j) = sum;
    }
#ifdef VERBOSE
    printf("state estimated\n");
#endif
    return estimation;
}

EstimationMean::~EstimationMean(){

}

EstimationMean::EstimationMean(){

}
