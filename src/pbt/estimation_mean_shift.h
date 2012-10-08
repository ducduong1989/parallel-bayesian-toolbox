#ifndef MEAN_SHIFT_ESTIMATION
#define MEAN_SHIFT_ESTIMATION

#include "estimation.h"

class EstimationMeanShift : public Estimation {
private:
        unsigned int numClasses;
        unsigned int maxIters;
        float dist;
        float convergenceDistance;

        ucolvec assign;
        fmat centers;
        ucolvec centerCounts;
        frowvec centerWeights;
        fvec reference;

        Particles* particles;

public:
        fmat getEstimation(Particles*);
        void setConfiguration(float distance, float epsilon, unsigned int maxIterations);
        unsigned int getClosestExistingCenter();
        void setRefereceVector(fvec);

        EstimationMeanShift();
        ~EstimationMeanShift();
};

#endif 
