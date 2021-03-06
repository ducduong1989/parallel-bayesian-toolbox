#include "resampling_multinomial_cuda.h"

Particles ResamplingMultinomialCUDA::resample(Particles* particles){

    Particles resampledSet;
    resampledSet.samples = fmat(particles->samples.n_rows,particles->samples.n_cols);
    resampledSet.weights = frowvec(particles->weights.n_cols);

    fvec cumsum = zeros<fvec>(particles->weights.n_cols);
    fvec random = randu<fvec>(particles->weights.n_cols);

    for (unsigned int i=1; i < particles->weights.n_cols;++i){
        cumsum(i) = cumsum(i-1) + particles->weights(i);
    }

    random=random * cumsum(cumsum.n_rows-1);

    // sort random
    random = sort(random);

    for (unsigned int j=0; j < random.n_rows; ++j){
        for (unsigned int i=0 ; i < cumsum.n_rows; ++i){
            if (random(j) <= cumsum(i)){
                if(i > 0){
                    if(random(j) >= cumsum(i-1)) {
                        for (unsigned int k=0;k<particles->samples.n_rows;++k){
                            resampledSet.samples(k,j) = particles->samples(k,i);
                        }
                        break;
                    }
                }
                else {
                    for (unsigned int k=0;k<particles->samples.n_rows;++k){
                        resampledSet.samples(k,j) = particles->samples(k,i);
                    }
                    break;

                }

            }
            // Normalize weights
            resampledSet.weights(j) = 1.0f/particles->weights.n_cols;
        }
    }

    return resampledSet;
}
