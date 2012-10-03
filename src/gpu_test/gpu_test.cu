#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

__global__ void cartToPolarTransformation(float* x_dev, int numberOfSamples, float* result_dev)
{
    // old version without threads
    //int dimension = blockIdx.y;
    //int sample = blockIdx.x;

    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/2) + threadIdx.x;
    if (sample < numberOfSamples){
        int offset = sample * 2;
        if (dimension == 0)
        {
            result_dev[offset] = sqrt(x_dev[offset] * x_dev[offset] + x_dev[offset+1]*x_dev[offset+1]);

        }
        if (dimension == 1)
        {
            result_dev[offset+1] = atan2(x_dev[offset+1], x_dev[offset]);
        }
    }
}


__global__ void polToCartTransformation(float* x_dev, int numberOfSamples, float* result_dev)
{
    // old version without threads
    //int dimension = blockIdx.y;
    //int sample = blockIdx.x;

    int dimension = threadIdx.y;
    int sample = blockIdx.x*(MAX_THREADS_PER_BLOCK/2) + threadIdx.x;
    if (sample < numberOfSamples){
        int offset = sample * 2;
        if (dimension == 0)
        {
            result_dev[offset] = x_dev[offset] * cos(x_dev[offset+1]);

        }
        if (dimension == 1)
        {
            result_dev[offset+1] = x_dev[offset] * sin(x_dev[offset+1]);
        }
    }
}


void callCartToPolarTransformation(float* devX, int dim, int num,float* devResult)
{
    //dim3 grid(num,dim);
    //cartToPolarTransformation<<< grid , 1 >>>(devX, num, devResult);


    int numberOfBlocks = num/(MAX_THREADS_PER_BLOCK/dim);
    if (num%(MAX_THREADS_PER_BLOCK/dim))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/dim,dim);
    cartToPolarTransformation<<< blockGrid , threadGrid >>>(devX, num, devResult);
}

void callPolarToCartTransformation(float* devX, int dim, int num,float* devResult)
{
    //dim3 grid(num,dim);
    //polToCartTransformation<<< grid , 1 >>>(devX, num, devResult);

    int numberOfBlocks = num/(MAX_THREADS_PER_BLOCK/dim);
    if (num%(MAX_THREADS_PER_BLOCK/dim))
    {
        numberOfBlocks++;
    }
    dim3 blockGrid(numberOfBlocks ,1);
    dim3 threadGrid(MAX_THREADS_PER_BLOCK/dim,dim);
    polToCartTransformation<<< blockGrid , threadGrid >>>(devX, num, devResult);
}
