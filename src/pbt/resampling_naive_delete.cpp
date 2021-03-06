#include "resampling_naive_delete.h"
#include <stdlib.h>
#include <stdio.h>


Particles ResamplingNaiveDelete::resample(Particles* particles){

    Particles resampledSet;

    if (particles->samples.n_cols>=decreaseThreshold)
    {
        // count particles tobe in resampled set
        unsigned int numberOfResampledParticles = 0;
        for (unsigned int i=0;i<particles->weights.n_cols;++i)
        {
            if (particles->weights(i) >= increaseThreshold)
            {
                numberOfResampledParticles++;
            }
        }

        // allocate matricies
        resampledSet.samples = fmat(particles->samples.n_rows,numberOfResampledParticles);
        resampledSet.weights = frowvec(numberOfResampledParticles);

        // save data in new matricies
        unsigned int indexInNewMatrix=0;
        for (unsigned int i=0;i<particles->weights.n_cols;++i)
        {
            if (particles->weights(i) >= increaseThreshold)
            {
                for (unsigned int k=0;k<particles->samples.n_rows;++k){
                    resampledSet.samples(k,indexInNewMatrix) = particles->samples(k,i);
                }
                resampledSet.weights(indexInNewMatrix) = particles->weights(i);
                indexInNewMatrix ++;
            }
        }

    }
    else
    {
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
    }


    return resampledSet;
}

void ResamplingNaiveDelete::setThresholds(float decrease, unsigned int increase){
    decreaseThreshold = decrease;
    increaseThreshold = increase;
}

ResamplingNaiveDelete::ResamplingNaiveDelete(){
    increaseThreshold = 100;
    decreaseThreshold = 0.01f;
}

ResamplingNaiveDelete::~ResamplingNaiveDelete(){

}


