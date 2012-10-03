#ifndef __MODEL_POINT__
#define __MODEL_POINT__

#include "model.h"

class ModelPoint3D: public Model{
public:
    ModelPoint3D();
    ~ModelPoint3D();
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);
};

#endif
