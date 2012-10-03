#ifndef __MODEL_MULTIMAP_SLAM__
#define __MODEL_MULTIMAP_SLAM__

#include "model.h"

class ModelCarMultimapSLAM: public Model{
private:
    fmat mapRight;
    fmat mapCenter;
    fmat mapLeft;

    fmat hitsRight;
    fmat missesRight;
    fmat hitsCenter;
    fmat missesCenter;
    fmat hitsLeft;
    fmat missesLeft;

    fmat lastState;

    fmat splineRight;
    fmat splineCenter;
    fmat splineLeft;

    unsigned int viewX;
    unsigned int viewY;

    unsigned int zeroX;
    unsigned int zeroY;

    unsigned int mapCellsX;
    unsigned int mapCellsY;
    float deltaT; ///< time difference between two measurements, e.g. cycle time

    fmat doLocalizationMeasurements(fmat *state);
    fmat generateCamView(fmat *spline);
    fmat applyGaussianNoise(fmat* input);
    void updateHitsAndMisses(fmat* state, fmat* camView, fmat* hits, fmat* misses);
public:
    float spacingX;
    float spacingY;
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

    ModelCarMultimapSLAM();
    ~ModelCarMultimapSLAM();

    //void showCam(fmat* cam);
};

#endif
