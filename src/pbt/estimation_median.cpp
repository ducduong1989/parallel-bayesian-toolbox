#include "estimation_median.h"
#include <stdlib.h>

fmat EstimationMedian::getEstimation(Particles* input){
    //float sum=0.0;
    //float weightSum = 0.0;
    fvec  estimation(input->samples.n_rows);

    /*uvec sortedIndex;

#ifdef VERBOSE
        printf("calculating estimation of %d particles in %d dimensions\n",input.samples.n_cols,input.samples.n_rows);
#endif

    sortedIndex = sort_index(input.weights);

    for (unsigned int j=0;j<input.samples.n_rows;++j){
            sum = 0.0;
            weightSum=0.0;
            for (unsigned int i=0; i<input.samples.n_cols; ++i){
                    sum += input.samples(j,i) * input.weights(i);
                    weightSum += input.weights(i);
                    if (weightSum >= 0.5) break;
            }
            estimation(j) = sum / weightSum;
    }*/

    fmat sortedSet = sort(input->samples,0,1);

    for (unsigned int i=0; i< input->samples.n_rows;++i){
        estimation(i) = sortedSet(i,sortedSet.n_cols/2);
    }

    return estimation;
}

EstimationMedian::~EstimationMedian(){

}

EstimationMedian::EstimationMedian(){

}
