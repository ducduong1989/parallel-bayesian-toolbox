#ifndef __NOISE_EXP__
#define __NOISE_EXP__


#include "noise.h"
#include <cmath>

class NoiseExponential : public Noise {
    public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseExponential(float);
    ~NoiseExponential();
};

#endif
