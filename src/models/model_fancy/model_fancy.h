#ifndef __MODEL_FANCY__
#define __MODEL_FANCY__

#include "model.h"

class ModelFancy: public Model
{
public:
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);

};

#endif
