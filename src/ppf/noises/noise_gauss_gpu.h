#ifndef __NOISE_GAUSS__
#define __NOISE_GAUSS__

#include "noise.h"
#include <cmath>

class NoiseGaussianGPU : public Noise {
public:
    frowvec eval(frowvec);
	frowvec eval(float* input, int number, int dim, int numberOfDims);
    frowvec sample(unsigned int);

    NoiseGaussianGPU(float,float);
     NoiseGaussianGPU(const NoiseGaussianGPU&);
    ~NoiseGaussianGPU();
};

#endif
