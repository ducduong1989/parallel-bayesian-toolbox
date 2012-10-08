#include "particle.h"


Particle::Particle(int dimension){
	dim = dimension;
	value = new float[dimension];
}

Particle::Particle(){
	dim = 1;
}

Particle::~Particle(){
        delete value;
}

Particle::Particle(const Particle& other)
{
	weight = other.weight;
	dim= other.dim;
	value = new float[other.dim];

	for (int i=0;i<dim;++i){
		value[i]=other.value[i];
	}

}

Particle &Particle::operator =(const Particle & other)
{
	weight = other.weight;
	dim = other.dim;
	value = new float[other.dim];

	for (int i=0;i<dim;++i){
		value[i]=other.value[i];
	}
	return *this;
}

bool Particle::operator < (const Particle & rhs ) const { return weight < rhs.weight; }

