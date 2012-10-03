#ifndef __MODEL_LOCALIZATION__
#define __MODEL_LOCALIZATION__

#include "model.h"

class ModelCarLocalization: public Model{
private:
    fmat mapRight;
    fmat mapCenter;
    fmat mapLeft;
    //fmat hits;
    //fmat misses;
    fmat lastState;
    fmat splineRight;
    fmat splineCenter;
    fmat splineLeft;

    unsigned int mapCellsX;
    unsigned int mapCellsY;
    float deltaT; ///< time difference between two measurements, e.g. cycle time
public:
    float spacingX;
    float spacingY;
    unsigned int zeroCellX;
    unsigned int zeroCellY;
    fvec speed;

    void setDescriptionTag();

    frowvec eval(fmat* values);

    void setPNoise();

    void setONoise();

    fmat ffun(fmat* current);

    fmat hfun(fmat* state);

    fmat getMap();

    void setMapRight(fmat);
    void setMapCenter(fmat);
    void setMapLeft(fmat);

    void setRightSpline(fmat);
    void setCenterSpline(fmat);
    void setLeftSpline(fmat);

    ModelCarLocalization();
    ~ModelCarLocalization();

};

#endif
