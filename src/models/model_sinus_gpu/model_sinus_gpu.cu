#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

/**
  * GPU based ffun
  */
extern "C" __global__ void pbtWPAMffunCUDA(float* lastState_dev, float* pNoise_dev, int stateDimension,
                                           int numberOfSamples,float* newState_dev)
{
    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/stateDimension) + threadIdx.x;
    if (dimension < stateDimension){
        if (sample < numberOfSamples)
        {
            if (dimension == 0){
                newState_dev[dimension+sample*stateDimension] = lastState_dev[dimension+sample*stateDimension]+0.1f;
            }
            if (dimension == 1){
                newState_dev[dimension+sample*stateDimension] = sin(newState_dev[dimension+sample*stateDimension]);
            }

            // attention! state matricies in column major format and random number matricies in row major format
            newState_dev[dimension+sample*stateDimension] += pNoise_dev[sample + numberOfSamples * dimension];
        }
    }
}

void callFfunKernel(float* lastState_dev, float* pNoise_dev, int stateDimension,
                    int numberOfSamples,float* newState_dev)
{
    int numberOfBlocks = numberOfSamples/(MAX_THREADS_PER_BLOCK/stateDimension);
    if (numberOfSamples%(MAX_THREADS_PER_BLOCK/stateDimension))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/stateDimension,stateDimension);
    pbtWPAMffunCUDA<<< blockGrid, threadGrid >>>(lastState_dev, pNoise_dev, stateDimension,numberOfSamples,newState_dev);
}

extern "C" __global__ void pbtWPAMhfunCUDA(float* state_dev, float* oNoise_dev, int stateDimension,
                                           int numberOfSamples, float* meas_dev)
{
    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/stateDimension) + threadIdx.x;
    if (dimension < stateDimension){
        if (sample < numberOfSamples)
        {
            // attention! state matricies in column major format and random number matricies in row major format
            meas_dev[dimension+sample*stateDimension] = state_dev[dimension+sample*stateDimension] + oNoise_dev[sample + numberOfSamples * dimension];
        }
    }
}

void callHfunKernel(float* state_dev, float* oNoise_dev, int stateDimension,
                    int numberOfSamples, float* meas_dev)
{
    int numberOfBlocks = numberOfSamples/(MAX_THREADS_PER_BLOCK/stateDimension);
    if (numberOfSamples%(MAX_THREADS_PER_BLOCK/stateDimension))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/stateDimension,stateDimension);
    pbtWPAMhfunCUDA<<< blockGrid, threadGrid >>>(state_dev,oNoise_dev,stateDimension,numberOfSamples,meas_dev);
}
