#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

/**
  * GPU based mean calculation
  */
extern "C" __global__ void ppfCalcDeviationsCUDA(float* virtualMeasurements, float* realMeasurement,
                                                 int stateDimension, int numberOfSamples, float* deviations){
    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/stateDimension) + threadIdx.x;
    if (dimension < stateDimension){
        if (sample < numberOfSamples)
        {
            deviations[dimension+sample*dimension] = virtualMeasurements[dimension+sample*dimension]
                                                     - realMeasurement[dimension];
        }
    }
}

/**
  * Call function for cuda kernel which calculates the deviations of virtual measurements
  */
void callDeviationKernel(float* virtualMeasurements, float* realMeasurement,
                         int stateDimension, int numberOfSamples, float* deviations){

    int numberOfBlocks = numberOfSamples/(MAX_THREADS_PER_BLOCK/stateDimension);
    if (numberOfSamples%(MAX_THREADS_PER_BLOCK/stateDimension))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/stateDimension,stateDimension);
    //call kernel
    ppfCalcDeviationsCUDA<<<blockGrid,threadGrid>>>(virtualMeasurements, realMeasurement,
                                                    stateDimension, numberOfSamples, deviations);
}
