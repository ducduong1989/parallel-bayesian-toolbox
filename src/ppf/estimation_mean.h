#ifndef MEAN_ESTIMATION
#define MEAN_ESTIMATION

#include "estimation.h"

class EstimationMean : public Estimation {
private:
	int dim;
public:
        fmat getEstimation(Particles*);

        EstimationMean();
        ~EstimationMean();
};

#endif 
