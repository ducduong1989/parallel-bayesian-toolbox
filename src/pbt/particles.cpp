#include "particles.h"

Particles::Particles(){
	dim = 1;
        samples = fmat(1,1);
        weights = frowvec(1);
}

Particles::~Particles(){

}

Particles::Particles(const Particles& other)
{
	weights = other.weights;
    dim= other.dim;

	samples = other.samples;
}

Particles &Particles::operator =(const Particles & other)
{
    weights = other.weights;
    dim= other.dim;

    samples = other.samples;

	return *this;
}

