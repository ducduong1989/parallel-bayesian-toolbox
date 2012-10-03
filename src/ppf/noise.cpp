#include "noise.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

Noise::Noise(){
}

frowvec Noise::sample(unsigned int number){

    frowvec sampleOutput = zeros<frowvec>(number);

    return sampleOutput;
}

frowvec Noise::eval(frowvec x){
    frowvec result = zeros<frowvec>(x.n_cols);

    return result;
}

frowvec Noise::eval(float* input, int number, int dim, int numberOfDims){
    frowvec result = zeros<frowvec>(1);

    return result;
}

Noise::~Noise(){

}

Noise::Noise(const Noise& other)
{
	a = other.a;
	b = other.b;
	c = other.c;
}

Noise &Noise::operator =(const Noise & other)
{
	a = other.a;
	b = other.b;
	c = other.c;

	return *this;
}
