#ifndef MODEL_CUDA
#define MODEL_CUDA
#include "model.h"
#include "noise_batch.h"
#include <armadillo>
#include <string>

using namespace arma;

class ModelCUDA : public Model{

public:
    //std::string descriptionTag;
    virtual void setPNoise();
    virtual void setONoise();

    //NoiseBatch pNoise;	//*< Batch of process noises for each coordinate of state vector
    //NoiseBatch oNoise;	//*< Batch of measurement noises for each coordinate of state vector

    virtual fmat ffun(fmat*);
    virtual float* ffun_gpu(fmat* current);

    virtual fmat hfun(fmat*);
    virtual float* hfun_gpu(float* values, int numberOfParticles, int stateDimension);

    ModelCUDA();

    virtual void setDescriptionTag();

    virtual frowvec eval(fmat* values);
    virtual frowvec eval_gpu(float* values, int numberOfParticles);

    //unsigned int getProcessDimension();
    //unsigned int getMeasurementDimension();

    //void initialize();

    //NoiseBatch getProcessNoise();
    //NoiseBatch getObservationNoise();

    //void setProcessNoise(NoiseBatch newPNoise);
    //void setObservationNoise(NoiseBatch newONoise);
};

#endif
