#ifndef MULTINOMIAL_RESAMPLING_CUDA
#define MULTINOMIAL_RESAMPLING_CUDA

#include "resampling.h"

void callMultinomialResamlingKernel(float* oldsamples, float* cumsum, float* random,
                                    unsigned int stateDimension, unsigned int numberOfSamples,
                                    float* newsamples);

class ResamplingMultinomialCUDA : public Resampling{
private:

public:
    ResamplingMultinomialCUDA();
    ~ResamplingMultinomialCUDA();

    Particles resample(Particles*);

};

#endif 
