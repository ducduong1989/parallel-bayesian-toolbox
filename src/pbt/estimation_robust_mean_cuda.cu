/**
  * GPU based robust mean calculation
  */
extern "C" __global__ void pbtCalcRobustMeanCUDA(float* data, float* weights, float* distances,
                                           int stateDimension, int numberOfSamples, int mostProbParticle,
                                           float* estimation){
    int dimension = blockIdx.x;
    if (dimension < stateDimension){
        estimation[dimension] = 0;
        for (unsigned int i = 0; i < numberOfSamples; ++i){
            if(fabsf(data[dimension + (i*stateDimension)] - data[dimension + (mostProbParticle*stateDimension)]) > distances[dimension]) continue;
            estimation[dimension] += data[dimension + (i*stateDimension)] * weights[dimension];
        }
    }
}


void callCalcRobustMeanKernel(float* data, float* weights, float* distances,
                        int stateDimension, int numberOfSamples, int mostProbParticle,
                        float* estimation){
    //call kernel
    pbtCalcRobustMeanCUDA<<<stateDimension,1>>>(data, weights, distances,stateDimension,
                                          numberOfSamples, mostProbParticle,estimation);
}
