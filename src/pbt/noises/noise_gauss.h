#ifndef __NOISE_GAUSS__
#define __NOISE_GAUSS__

#include "noise.h"
#include <cmath>

class NoiseGaussian : public Noise {
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseGaussian(float,float);
     NoiseGaussian(const NoiseGaussian&);
    ~NoiseGaussian();
};

#endif
