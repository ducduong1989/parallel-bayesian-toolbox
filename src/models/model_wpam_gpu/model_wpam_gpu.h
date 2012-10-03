#ifndef __MODEL_WPAM_GPU__
#define __MODEL_WPAM_GPU__

#include "model.h"

#include <curand.h>

void callFfunKernel(float* lastState_dev, float* F_dev, float* U_dev, float* pNoise_dev, int stateDimension, int numberOfSamples,float* newState_dev);
void callHfunKernel(float* state_dev, float* oNoise_dev, int stateDimension, int numberOfSamples, float* meas_dev);

//class
class ModelWPAMGPU: public Model{
private:
	curandGenerator_t gen;
    fmat F;
    float T;
    float variance;
    unsigned int dimmeas;

    NoiseBatch U;

public:
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    void setUNoise();

    fmat ffun(fmat* current);
    float* ffun_gpu(fmat* current);

    fmat hfun(fmat* state);
	float* hfun_gpu(float* values, int numberOfParticles, int stateDimension);

    void addF();

	frowvec eval_gpu(float* values, int numberOfParticles);

    ModelWPAMGPU();
    ~ModelWPAMGPU();

};


#endif
