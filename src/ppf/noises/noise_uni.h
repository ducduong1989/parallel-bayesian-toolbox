#ifndef __NOISE_UNI__
#define __NOISE_UNI__


#include "noise.h"
#include <cmath>

class NoiseUniform : public Noise {
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseUniform(float,float);
    ~NoiseUniform();
};

#endif
