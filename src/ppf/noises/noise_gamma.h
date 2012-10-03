#ifndef __NOISE_GAMMA__
#define __NOISE_GAMMA__


#include "noise.h"
#include <cmath>

class NoiseGamma : public Noise {
private:
    long factorial(unsigned int);
	unsigned int q;
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseGamma(unsigned int);
    ~NoiseGamma();
};

#endif
