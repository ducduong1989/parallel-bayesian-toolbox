#ifndef __MODEL_LINEAR__
#define __MODEL_LINEAR__

#include "model.h"

class ModelLinear: public Model{
public:
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);

};

#endif
