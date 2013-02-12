#ifndef MEAN_ESTIMATION_CUDA
#define MEAN_ESTIMATION_CUDA

#include "estimation.h"

void callCalcRobustMeanKernel(float* data, float* weights, float* distances,
                              int stateDimension, int numberOfSamples, int mostProbParticle,
                              float* estimation);

/**
  * GPU accelerated robust mean estimation
  */
class EstimationRobustMeanCUDA : public Estimation {
private:
        fvec distances;
public:
        fmat getEstimation(Particles*);
        void setConfiguration(fvec);

        EstimationRobustMeanCUDA();
        ~EstimationRobustMeanCUDA();
};

#endif 
