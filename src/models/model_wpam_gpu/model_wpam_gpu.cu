#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

/**
  * GPU based ffun
  */
extern "C" __global__ void ppfWPAMffunCUDA(float* lastState_dev, float* F_dev, float* U_dev, float* pNoise_dev, int stateDimension, int numberOfSamples,float* newState_dev)
{
    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/9) + threadIdx.x;
    if (dimension < stateDimension){
        if (sample < numberOfSamples)
        {
            newState_dev[dimension+sample*stateDimension] = 0;
            // attention! state matricies in column major format and random number matricies in row major format
            for (unsigned int i = 0; i < 9; ++i)
            {
                newState_dev[dimension+sample*dimension] += F_dev[dimension + i*stateDimension] * lastState_dev[sample*stateDimension+i];
            }
            newState_dev[dimension+sample*stateDimension] += U_dev[sample + numberOfSamples * dimension];
            newState_dev[dimension+sample*stateDimension] += pNoise_dev[sample + numberOfSamples * dimension];
        }
    }
}

void callFfunKernel(float* lastState_dev, float* F_dev, float* U_dev, float* pNoise_dev, int stateDimension, int numberOfSamples,float* newState_dev)
{
    int numberOfBlocks = numberOfSamples/(MAX_THREADS_PER_BLOCK/stateDimension);
    if (numberOfSamples%(MAX_THREADS_PER_BLOCK/stateDimension))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/stateDimension,stateDimension);
    ppfWPAMffunCUDA<<< blockGrid, threadGrid >>>(lastState_dev,F_dev,U_dev, pNoise_dev, stateDimension,numberOfSamples,newState_dev);
}

extern "C" __global__ void ppfWPAMhfunCUDA(float* state_dev, float* oNoise_dev, int stateDimension, int numberOfSamples, float* meas_dev)
{
    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/9) + threadIdx.x;
    if (dimension < stateDimension){
        if (sample < numberOfSamples)
        {
            // attention! state matricies in column major format and random number matricies in row major format
            meas_dev[dimension+sample*stateDimension] = state_dev[dimension+sample*stateDimension] + oNoise_dev[sample + numberOfSamples * dimension];
        }
    }
}

void callHfunKernel(float* state_dev, float* oNoise_dev, int stateDimension, int numberOfSamples, float* meas_dev)
{
    int numberOfBlocks = numberOfSamples/(MAX_THREADS_PER_BLOCK/stateDimension);
    if (numberOfSamples%(MAX_THREADS_PER_BLOCK/stateDimension))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/stateDimension,stateDimension);
    ppfWPAMhfunCUDA<<< blockGrid, threadGrid >>>(state_dev,oNoise_dev,stateDimension,numberOfSamples,meas_dev);
}
