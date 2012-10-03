#include "model_sinus_gpu.h"
#include <math.h>
#include "noise_gauss.h"

#include <cuda.h>
#include <cuda_runtime.h>

ModelSinusGPU::ModelSinusGPU()
{
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

ModelSinusGPU::~ModelSinusGPU()
{

}

void ModelSinusGPU::setDescriptionTag()
{
    descriptionTag = "Sinus signal model";
}

void ModelSinusGPU::setPNoise()
{
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.05f);
    pNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.05f);
    pNoise.addNoise(y);
}

void ModelSinusGPU::setONoise()
{
    NoiseGaussian* x = new NoiseGaussian( 0.0f, 0.01f);
    oNoise.addNoise(x);
    NoiseGaussian* y = new NoiseGaussian( 0.0f, 0.01f);
    oNoise.addNoise(y);
}

fmat ModelSinusGPU::ffun(fmat *current)
{

    fmat prediction(current->n_rows,current->n_cols);
    fmat pNoiseSample = pNoise.sample(current->n_cols);

    for (unsigned int i=0;i<current->n_cols;++i){
        prediction(0,i) = current->at(0,i)+0.1f;

        prediction(1,i) = sin(prediction(0,i));
    }

    prediction = prediction + pNoiseSample;

    return prediction;
}

float* ModelSinusGPU::ffun_gpu(fmat *current)
{
    float* lastState_dev;
    float* pNoise_dev;
    int stateDimension = current->n_rows;
    int numberOfSamples = current->n_cols;
    float* newState_dev;

    //allocate memory on gpu
    cudaMalloc( &lastState_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;
    cudaMalloc( &pNoise_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;
    cudaMalloc( &newState_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;

    //Copy particles and weights to the gpu
    cudaMemcpy(lastState_dev,current->memptr(),(size_t) current->n_elem * sizeof(float), cudaMemcpyHostToDevice);

    //pNoise
    curandGenerateNormal(gen, pNoise_dev, numberOfSamples, pNoise.batch.at(0)->a, pNoise.batch.at(0)->b);
    curandGenerateNormal(gen, pNoise_dev+numberOfSamples, numberOfSamples, pNoise.batch.at(1)->a, pNoise.batch.at(1)->b);

    //prediction = F * current + pNoiseSample + u ;
    callFfunKernel(lastState_dev, pNoise_dev, stateDimension ,numberOfSamples,newState_dev);
    //printf("ffun error: %s\n",cudaGetErrorString(cudaGetLastError()));

    //get estimation from gpu
    //cudaMemcpy(prediction.memptr(),newState_dev,current.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

    // clean up the graphics card
    cudaFree(lastState_dev);
    cudaFree(pNoise_dev);

    return newState_dev;
}

fmat ModelSinusGPU::hfun(fmat *state)
{

    fmat measurement(state->n_rows,state->n_cols);
    fmat oNoiseSample = oNoise.sample(state->n_cols);

    measurement = *state + oNoiseSample;

    return measurement;
}

float* ModelSinusGPU::hfun_gpu(float* values, int numberOfSamples, int stateDimension)
{

    float* oNoise_dev;
    float* meas_dev;

    //allocate memory on gpu
    cudaMalloc( &oNoise_dev, (size_t) numberOfSamples * stateDimension * sizeof(float)) ;
    cudaMalloc( &meas_dev, (size_t) numberOfSamples * stateDimension * sizeof(float)) ;

    //generate random particles
    curandGenerateNormal(gen, oNoise_dev, numberOfSamples, oNoise.batch.at(0)->a, oNoise.batch.at(0)->b);
    curandGenerateNormal(gen, oNoise_dev+numberOfSamples, numberOfSamples, oNoise.batch.at(1)->a, oNoise.batch.at(0)->b);

    //prediction = F * current + pNoiseSample + u ;
    callHfunKernel(values,oNoise_dev,stateDimension,numberOfSamples,meas_dev);

    // clean up the graphics card
    cudaFree(oNoise_dev);
    cudaFree(values);

    return meas_dev;
}

