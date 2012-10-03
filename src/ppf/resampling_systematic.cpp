#include "resampling_systematic.h"
#include <stdlib.h>
#include <stdio.h>


Particles ResamplingSystematic::resample(Particles* particles){
    bool sortingNecessary = false;

    Particles resampledSet;
    resampledSet.samples = fmat(particles->samples.n_rows,particles->samples.n_cols);
    resampledSet.weights = frowvec(particles->weights.n_cols);

    fvec cumsum = zeros<fvec>(particles->samples.n_cols);
    fvec random = randu<fvec>(1);
    fvec targetRandom = zeros<fvec>(particles->samples.n_cols);
    float randomWidth = (float)1.0/particles->samples.n_cols;

    for (unsigned int i=1; i < particles->weights.n_cols;++i){
        cumsum(i) = cumsum(i-1) + particles->weights(i);
    }

    // generate one random number for startpoint in the sampling circle
    random=random/randomWidth;

    targetRandom(0) = random(0);
    for (unsigned int i=1; i < random.n_rows;++i){
        targetRandom(i) = targetRandom(0)+i*randomWidth;
        if (targetRandom(i)>=1){
            targetRandom(i) = targetRandom(i)-1;
            sortingNecessary = true;
        }
    }

    if (sortingNecessary){
        targetRandom = sort(targetRandom);
    }

    targetRandom = targetRandom * cumsum(cumsum.n_rows-1);

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

ResamplingSystematic::ResamplingSystematic(){

}

ResamplingSystematic::~ResamplingSystematic(){

}
