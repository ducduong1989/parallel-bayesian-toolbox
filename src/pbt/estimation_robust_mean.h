#ifndef MEAN_ESTIMATION
#define MEAN_ESTIMATION

#include "estimation.h"

/**
  * robust mean estimation
  */
class EstimationRobustMean : public Estimation {
private:
        fvec distances;
public:
        fmat getEstimation(Particles*);
        void setConfiguration(fvec);

        EstimationRobustMean();
        ~EstimationRobustMean();
};

#endif 
