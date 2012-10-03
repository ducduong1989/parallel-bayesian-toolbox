#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

/**
  * GPU based multinomial resampling
  */
extern "C" __global__ void ppfMultinomialResamplingCUDA(float* oldsamples, float* weights, float* random,
                                                        unsigned int stateDimension, unsigned int numberOfSamples,
                                                        float* newsamples){

    int workParticle = blockIdx.x*(MAX_THREADS_PER_BLOCK) + threadIdx.x;
    int takeFromNumber = 0;
    float randomNumber = 0;
    float sum=0;
    unsigned int i,j;
    if (workParticle < numberOfSamples){
        //search corresponding particle
        randomNumber = random[workParticle];

        for (j=0; j< numberOfSamples; ++j){
            sum = sum + weights[j];
            if (sum >= randomNumber){
                takeFromNumber = j;
                break;
            }
        }

        // save particles in new particle set
        for (i=0; i< stateDimension; ++i){
            newsamples[i+workParticle*stateDimension] = oldsamples[i+takeFromNumber*stateDimension];
        }
    }
}


void callMultinomialResamlingKernel(float* oldsamples, float* weights, float* random,
                                    unsigned int stateDimension, unsigned int numberOfSamples,
                                    float* newsamples)
{
    int numberOfBlocks = numberOfSamples/(MAX_THREADS_PER_BLOCK);
    if (numberOfSamples%(MAX_THREADS_PER_BLOCK))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK,1);

    ppfMultinomialResamplingCUDA<<< blockGrid , threadGrid >>>(oldsamples, weights, random, stateDimension,
                                       numberOfSamples, newsamples);
}
