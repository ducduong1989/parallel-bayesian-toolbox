#include "noise_studentt.h"
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265
#endif


frowvec NoiseStudentT::sample(unsigned int number){
    frowvec samples;

    samples = a + randn<frowvec>(number)*b;

    return samples;
}

frowvec NoiseStudentT::eval(frowvec input){
    frowvec result;

    result = (1/2*(float)sqrt(2*M_PI))*exp(-0.5f*((input-a)/b)%((input-a)/b));

    return result;
}
