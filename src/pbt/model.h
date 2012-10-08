#ifndef MODEL
#define MODEL

#include "noise_batch.h"
#include <armadillo>
#include <string>

using namespace arma;

class Model{

public:
    std::string descriptionTag;
    virtual void setPNoise();
    virtual void setONoise();

    virtual void setH();
    virtual void setF();

    NoiseBatch pNoise;	//*< Batch of process noises for each coordinate of state vector
    NoiseBatch oNoise;	//*< Batch of measurement noises for each coordinate of state vector

    virtual fmat ffun(fmat*);
    virtual fmat hfun(fmat*);

    fmat A;
    fmat B;
    fmat C;
    fmat H;
    fmat F;
    fvec u;
    fmat Q;
    fmat W; // also knows as R

    Model();

    virtual void setDescriptionTag();

    virtual frowvec eval(fmat* values);

    unsigned int getProcessDimension();
    unsigned int getMeasurementDimension();

    void initialize();

    NoiseBatch getProcessNoise();
    NoiseBatch getObservationNoise();

    void setProcessNoise(NoiseBatch newPNoise);
    void setObservationNoise(NoiseBatch newONoise);
};

#endif
