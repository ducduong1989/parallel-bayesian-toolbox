#ifndef __MODEL_WPAM__
#define __MODEL_WPAM__

#include "model.h"

class ModelWPAM: public Model{
private:
    float T;
    float variance;
    unsigned int dimmeas;

    //NoiseBatch U;

public:
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    void setUNoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);

    void setF();
    void setH();
    void setQ();
    void setW();

    ModelWPAM();
    ~ModelWPAM();

};

#endif
