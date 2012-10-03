#ifndef __NOISE_TRI__
#define __NOISE_TRI__


#include "noise.h"
#include <cmath>

class NoiseTriangular : public Noise {
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseTriangular(float,float);
    ~NoiseTriangular();
};

#endif
