#ifndef __NOISE_U__
#define __NOISE_U__


#include "noise.h"
#include <cmath>

class NoiseU : public Noise {
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseU(float,float);
    ~NoiseU();
};

#endif
