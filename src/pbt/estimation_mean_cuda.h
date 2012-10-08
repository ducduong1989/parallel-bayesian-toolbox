#ifndef MEAN_ESTIMATION_CUDA
#define MEAN_ESTIMATION_CUDA

#include "estimation.h"

void callCalcMeanKernel(float* data, float* weights, int stateDimension, int numberOfSamples, float* estimation);

class EstimationMeanCUDA : public Estimation {
public:
        fmat getEstimation(Particles*);

        EstimationMeanCUDA();
        ~EstimationMeanCUDA();
};

#endif
