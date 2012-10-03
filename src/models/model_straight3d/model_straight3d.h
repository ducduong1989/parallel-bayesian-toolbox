#ifndef __MODEL_STRAIGHT__
#define __MODEL_STRAIGHT__

#include "model.h"

class ModelStraight3D: public Model{
private:
    fvec pointOfOrigin;
    fvec directionVector;

public:
    ModelStraight3D();
    ~ModelStraight3D();
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);

    void addStraightDefinition(fvec origin, fvec vector);

};

#endif
