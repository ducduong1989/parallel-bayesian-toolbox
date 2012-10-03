#ifndef __MODEL_POINT__
#define __MODEL_POINT__

#include "model.h"

class ModelCircle2D: public Model{
private:
    float radius;
    float phiStep;
    fvec orig;

public:
    ModelCircle2D();
    ~ModelCircle2D();
    void setDescriptionTag();

    void setPNoise();

    void setONoise();

    void setRadius(float circleRadius);
    void setPhiStep(float phi);
    void setOrigin(fvec origin);

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);
};

#endif
