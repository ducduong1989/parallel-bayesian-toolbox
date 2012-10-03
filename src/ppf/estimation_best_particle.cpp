#include "estimation_best_particle.h"
#include <iostream>

fmat EstimationBestParticle::getEstimation(Particles* input){
    float max = input->weights(0);
    unsigned int mostProbableParticle = 0;
    fvec  estimation(input->samples.n_rows);

#ifdef VERBOSE
        printf("calculating estimation of %d particles in %d dimensions\n",input->samples.n_cols,input->samples.n_rows);
#endif

    for (unsigned int j=1;j<input->weights.n_cols;++j){
        if (max < input->weights(j)){
            max = input->weights(j);
            mostProbableParticle = j;
        }
    }
    estimation = input->samples.row(mostProbableParticle);

#ifdef VERBOSE
        printf("state estimated");
#endif
    return estimation;
}

EstimationBestParticle::~EstimationBestParticle(){

}

EstimationBestParticle::EstimationBestParticle(){

}
