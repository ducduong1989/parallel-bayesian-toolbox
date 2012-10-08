/**
  * \class NoiseTrapezoidal This class provides a noise with trapeziodal distribution
  * \warning the evaluation doesn't work, because it needs four parameters
  * and base class provides only three
  */

#ifndef __NOISE_CTRAP__
#define __NOISE_CTRAP__


#include "noise.h"
#include <cmath>

class NoiseCTrapezoidal : public Noise {
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseCTrapezoidal(float,float,float,float);
    ~NoiseCTrapezoidal();
};

#endif
