#ifndef __MODEL_SINUS_GPU__
#define __MODEL_SINUS_GPU__

#include "model_cuda.h"
#include <curand.h>

void callFfunKernel(float* lastState_dev, float* pNoise_dev, int stateDimension,
                    int numberOfSamples,float* newState_dev);
void callHfunKernel(float* state_dev, float* oNoise_dev, int stateDimension,
                    int numberOfSamples, float* meas_dev);

class ModelSinusGPU: public ModelCUDA
{
private:
    curandGenerator_t gen;
public:
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);
    float* ffun_gpu(fmat* current);

    fmat hfun(fmat* state);
    float* hfun_gpu(float* values, int numberOfParticles, int stateDimension);

    frowvec eval_gpu(float* values, int numberOfParticles);

    ModelSinusGPU();

    ~ModelSinusGPU();
};

#endif
