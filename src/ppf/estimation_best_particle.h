#ifndef MEAN_ESTIMATION
#define MEAN_ESTIMATION

#include "estimation.h"

class EstimationBestParticle : public Estimation {

public:
        fmat getEstimation(Particles*);

        EstimationBestParticle();
        ~EstimationBestParticle();
};

#endif 
