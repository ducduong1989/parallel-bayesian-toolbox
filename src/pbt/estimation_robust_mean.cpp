#include "estimation_robust_mean.h"

fmat EstimationRobustMean::getEstimation(Particles* input){
    float max = input->weights(0);
    unsigned int mostProbableParticle = 0;
    fvec  estimation(input->samples.n_rows);
    float weightSum = 0;
    float sum = 0;

    if (distances.n_rows != input->samples.n_rows){
        perror("EstimationRobustMean configuration error: Dimension of configuration and input doesn't match");
    }

#ifdef VERBOSE
        printf("calculating estimation of %d particles in %d dimensions\n",input->samples.n_cols,input->samples.n_rows);
#endif

    for (unsigned int j=1;j<input->weights.n_cols;++j){
        if (max < input->weights(j)){
            max = input->weights(j);
            mostProbableParticle = j;
        }
    }

    for (unsigned int j=0;j<input->samples.n_rows;++j){
            sum = 0.0;
            weightSum = 0.0;

            for (unsigned int i=0; i<input->samples.n_cols; ++i){
                if(fabsf(input->samples(j,i)- input->samples(j,mostProbableParticle)) > distances(j)) continue;

                sum += input->samples(j,i) * input->weights(i);
                weightSum += input->weights(i);
            }
            estimation(j) = sum/ weightSum;
    }
#ifdef VERBOSE
        printf("state estimated");
#endif
    return estimation;
}

void EstimationRobustMean::setConfiguration(fvec configuration){
    distances = configuration;
}

EstimationRobustMean::~EstimationRobustMean(){

}

EstimationRobustMean::EstimationRobustMean(){
    distances = fvec(1);
}
