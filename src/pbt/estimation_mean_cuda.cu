/**
  * GPU based mean calculation
  */
extern "C" __global__ void ppfCalcMeanCUDA(float* data, float* weights, int stateDimension, int numberOfSamples, float* estimation){
    int dimension = blockIdx.x;
    if (dimension < stateDimension){
        estimation[dimension] = 0;
        for (unsigned int i = 0; i < numberOfSamples; ++i){
            estimation[dimension] += data[dimension + (i*stateDimension)] * weights[dimension];
        }
    }
}


void callCalcMeanKernel(float* data, float* weights, int stateDimension, int numberOfSamples, float* estimation){
    //call kernel
    ppfCalcMeanCUDA<<<stateDimension,1>>>(data,weights,stateDimension,numberOfSamples,estimation);
}
