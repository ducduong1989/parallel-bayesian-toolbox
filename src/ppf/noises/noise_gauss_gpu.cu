#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI (3.14159265f)
#endif
#include <stdio.h>

#define MAX_THREADS_PER_BLOCK 512

__global__ void gaussEvalGPUKernel(float* x_dev, int number, float mean, float stddev, float* result_dev)
{
    int element = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (element < number){
        float preterm = (float) (1.0f/(stddev*sqrt((float)(2*M_PI))));

        result_dev[element] = preterm * expf(-0.5f*(((x_dev[element]-mean)/stddev)*((x_dev[element]-mean)/stddev)));
    }
}


void callGaussEvaluationV(float* x, int number, float mean, float stddev, float* result)
{
    int numberOfBlocks = number/MAX_THREADS_PER_BLOCK;
    if (number%MAX_THREADS_PER_BLOCK)
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK,1);
    gaussEvalGPUKernel<<< blockGrid , threadGrid >>>(x, number, mean, stddev, result);
}

// in the case that a float vector is givenn to the evaluation function, it assumes that the data structur is a column-major matrix
//so the dimension (number of rows of matrix) must be known to calculate the index
__global__ void gaussEvalGPUKernel2(float* x_dev, int dim, int number, int numberOfDims, float mean, float stddev, float* result_dev)
{
    int element = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;

    if (element < number){
		int offset = (element*numberOfDims) + dim; 
        float preterm = (float) (1.0f/(stddev*sqrt((float)(2*M_PI))));

        result_dev[element] = preterm * expf(-0.5f*(((x_dev[offset]-mean)/stddev)*((x_dev[offset]-mean)/stddev)));
    }
}


void callGaussEvaluationM(float* x, int number, int dim, int numberOfDims, float mean, float stddev, float* result)
{
	//printf("Number: %i\n Dim: %i\n NumDims: %i\n", number, dim, numberOfDims);
    int numberOfBlocks = number/MAX_THREADS_PER_BLOCK;
    if (number%MAX_THREADS_PER_BLOCK)
    {
        numberOfBlocks++;
    }
    //printf("Blocks: %i\n", numberOfBlocks);
    
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK,1);
    gaussEvalGPUKernel2 <<< blockGrid , threadGrid >>> (x, dim, number, numberOfDims, mean, stddev, result);
    //printf("Kernel passed with %s\n",cudaGetErrorString(cudaGetLastError()));
}
