#include "resampling_residual.h"
#include <stdlib.h>
#include <stdio.h>


Particles ResamplingResidual::resample(Particles* particles){

    Particles resampledSet;
    Particles stage1;
    int count = 0;
    resampledSet.samples = fmat(particles->samples.n_rows,particles->samples.n_cols);
    resampledSet.weights = frowvec(particles->weights.n_cols);

    fvec cumsum;
    fvec random;

    unsigned int number = particles->samples.n_cols;
    unsigned int numberOfStage1 = 0;

    // Generating copie information of every particle
    ivec copies = zeros<ivec>(particles->samples.n_cols);
    for (unsigned int i=0;i<particles->samples.n_cols;++i){
        copies = (int) floor(number*particles->weights(i));
    }

    numberOfStage1 = sum(copies);

    stage1.samples =  fmat(particles->samples.n_rows,numberOfStage1);
    stage1.weights = frowvec(numberOfStage1);

    //Picking N_i = N*w_i copies of i-th particle
    for (unsigned int i=1; i < copies.n_rows;++i){
        for (int j=0;j<copies(i);++j){
            for (unsigned int k=0;k<particles->samples.n_rows;++k){
                stage1.samples(k,count) = particles->samples(k,i);
            }
            stage1.weights(count) = particles->weights(i) - (float)copies(i)/number;
            count++;
        }
    }

    // multinomial resampling with residuum weights w_i = w_i - N_i/N
    cumsum = zeros<fvec>(numberOfStage1);
    for (unsigned int i=1; i < stage1.weights.n_cols;++i){
        cumsum(i) = cumsum(i-1) + stage1.weights(i);
    }

    // generate sorted random set
    random = randu<fvec>(numberOfStage1);
    random=random*cumsum(cumsum.n_rows-1);
    random = sort(random);

    for (unsigned int j=0; j < random.n_rows; ++j){
        for (unsigned int i=0 ; i < cumsum.n_rows; ++i){
            if (random(j) <= cumsum(i)){
                if(i > 0){
                    if(random(j) >= cumsum(i-1)) {
                        for (unsigned int k=0;k<stage1.samples.n_rows;++k){
                            resampledSet.samples(k,j) = stage1.samples(k,i);
                        }
                        break;
                    }
                }
                else {
                    for (unsigned int k=0; k<stage1.samples.n_rows; ++k){
                        resampledSet.samples(k,j) = stage1.samples(k,i);
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

ResamplingResidual::ResamplingResidual(){

}

ResamplingResidual::~ResamplingResidual(){

}


