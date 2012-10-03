#include "resampling_stratified.h"
#include <stdlib.h>
#include <stdio.h>


Particles ResamplingStratified::resample(Particles* particles){

    Particles resampledSet;
    resampledSet.samples = fmat(particles->samples.n_rows,particles->samples.n_cols);
    resampledSet.weights = frowvec(particles->weights.n_cols);

    fvec cumsum = zeros<fvec>(particles->samples.n_cols);
    fvec random = randu<fvec>(particles->samples.n_cols);
    fvec targetRandom = zeros<fvec>(particles->samples.n_cols);
    float randomWidth = (float)1.0/particles->samples.n_cols;

    for (unsigned int i=1; i < particles->weights.n_cols;++i){
        cumsum(i) = cumsum(i-1) + particles->weights(i);
    }

    // scale random numbers (0,1) to (0,max(csum))
    /*for (int i=1;i<particles.weight.n_rows;++i){
                random(i)=random(i)*cumsum(cumsum.n_rows-1);
    }*/
    random=random/randomWidth;

    for (unsigned int i=0; i < random.n_rows;++i){
        targetRandom(i) = i*randomWidth + random(i);
    }

    targetRandom = targetRandom*cumsum(cumsum.n_rows-1);

    for (unsigned int j=0; j < targetRandom.n_rows; ++j){
        for (unsigned int i=0 ; i < cumsum.n_rows; ++i){
            if (targetRandom(j) <= cumsum(i)){
                if(i > 0){
                    if(targetRandom(j) >= cumsum(i-1)) {
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

ResamplingStratified::ResamplingStratified(){

}

ResamplingStratified::~ResamplingStratified(){

}
