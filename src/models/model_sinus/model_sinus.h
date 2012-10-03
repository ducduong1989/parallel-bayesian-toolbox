#ifndef __MODEL_SINUS__
#define __MODEL_SINUS__

#include "model.h"

class ModelSinus: public Model
{
public:
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);


};

#endif
