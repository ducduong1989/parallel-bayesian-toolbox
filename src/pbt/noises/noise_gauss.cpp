#include "noise_gauss.h"
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


frowvec NoiseGaussian::sample(unsigned int number){
    frowvec samples;

    samples = a + randn<frowvec>(number) *b ;

    return samples;
}

frowvec NoiseGaussian::eval(frowvec input){
    frowvec result;

    float preterm = (float) (1.0f/(b*sqrt(2.0f*M_PI)));

    result = preterm * exp(-0.5f*(((input-a)/b)%((input-a)/b)));

    return result;
}

NoiseGaussian::NoiseGaussian(float mean, float stddev){
    a = mean;
    b= stddev;
}

NoiseGaussian::NoiseGaussian(const NoiseGaussian &other){
    a = other.a;
    b= other.b;
}


NoiseGaussian::~NoiseGaussian(){

}
