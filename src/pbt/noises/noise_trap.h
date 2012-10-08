/**
  * \class NoiseTrapezoidal This class provides a noise with trapeziodal distribution
  * \warning the evaluation doesn't work, because it needs four parameters
  * and base class provides only three
  */

#ifndef __NOISE_TRAP__
#define __NOISE_TRAP__


#include "noise.h"
#include <cmath>

class NoiseTrapezoidal : public Noise {
private:
    float a1;
    float a2;
    float b1;
    float b2;
    float lambda1;
    float lambda2;
    float beta;
    float a;
    float b;
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseTrapezoidal(float,float,float,float);
    ~NoiseTrapezoidal();
};

#endif
