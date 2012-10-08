#include "noise_u.h"
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265
#endif


frowvec NoiseU::sample(unsigned int number){
    frowvec samples;

    samples = (a+b)/2.0f + ((b-a)/2.0f)*sin(2.0f*(float)M_PI*randu<frowvec>(number));

    return samples;
}

frowvec NoiseU::eval(frowvec input){
    frowvec result = zeros<frowvec>(input.n_cols);
    float bMinusASquared = (b-a)*(b-a);
    float diff;

    for (unsigned int i = 0; i<input.n_cols; ++i)
    {
        if (( a < input(i)) && (input(i) < b))
        {
            diff = (2*input(i) - a -b);
            result(i) = (float)(2/M_PI) * pow(  (float)bMinusASquared - diff*diff  , -0.5f );
        }
    }

    return result;
}

NoiseU::NoiseU(float leftBorder, float rightBorder){
    a = leftBorder;
    b= rightBorder;
}

NoiseU::~NoiseU(){

}
