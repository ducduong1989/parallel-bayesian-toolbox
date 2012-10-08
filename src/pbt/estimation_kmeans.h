#ifndef K_MEAN_ESTIMATION
#define K_MEAN_ESTIMATION

#include "estimation.h"

class EstimationKMeans : public Estimation {
private:
        unsigned int numClasses;
        unsigned int maxIters;

        ucolvec assign;
        fmat centers;
        ucolvec centerCounts;
        frowvec centerWeights;
        fvec reference;

        Particles* particles;

public:
        fmat getEstimation(Particles*);
        void setConfiguration(unsigned int, unsigned int);
        unsigned int getClosestCenter(unsigned int);
        unsigned int getClosestExistingCenter();
        void initCenters();
        void setRefereceVector(fvec);

        EstimationKMeans();
        ~EstimationKMeans();
};

#endif 
