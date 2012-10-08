#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_THREADS_PER_BLOCK 512

/**
  * sum up pairs of floats
  */
extern "C" __global__ void summingPairs(float* setOfPairs, int numberOfNumbers, float* sums)
{

    int pair = blockIdx.x*(MAX_THREADS_PER_BLOCK) + threadIdx.x;

    if (pair < numberOfNumbers/2){
        if ( pair == (numberOfNumbers/2) )
        {

        }
        else
        {
            sums[pair] = setOfPairs[pair*2 ] + setOfPairs[pair*2 + 1];
        }
    }
}


/**
  * scales all numbers in vector by factor
  */
extern "C" __global__ void scaleNumbers(float* numbers, int number, float scaleFactor, float* scaledNumbers)
{

    int i = blockIdx.x*(MAX_THREADS_PER_BLOCK) + threadIdx.x;

    if (i < number){
        scaledNumbers[i] = numbers[i] * scaleFactor;
    }
}

/**
  * scales all numbers in vector by factor
  */
extern "C" __global__ void scaleNumbersInPlace(float* numbers, int number, float scaleFactor)
{

    int i = blockIdx.x*(MAX_THREADS_PER_BLOCK) + threadIdx.x;

    if (i < number){
        numbers[i] = numbers[i] * scaleFactor;
    }
}

float sum(float* vector, int number)
{
    int currentNumber = number;
    float sum = 0;
    float* sums = NULL;
    float* setOfPairs;
    int numberOfBlocks = 0;

    dim3 blockGrid;
    dim3 threadGrid;

    while (currentNumber > 1)
    {
        numberOfBlocks = currentNumber/(MAX_THREADS_PER_BLOCK);
        if (currentNumber%(MAX_THREADS_PER_BLOCK))
        {
            numberOfBlocks++;
        }
        blockGrid = dim3(numberOfBlocks ,1);
        threadGrid = dim3(MAX_THREADS_PER_BLOCK,1);

        summingPairs<<< blockGrid, threadGrid >>>(setOfPairs, currentNumber, sums);
        setOfPairs = sums;

        currentNumber = currentNumber/2;
        if (number%2)
        {
            currentNumber++;
        }
    }

    return sum;
}


float max(float* vector, int number)
{
    float max = 0;

    return max;
}
