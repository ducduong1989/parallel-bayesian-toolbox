#include "noise_batch.h"

#include <cmath>
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14
#endif

using namespace arma;

fmat NoiseBatch::sample(unsigned int number){

    fmat samples((int)batch.size(),number);
    samples = batch.at(0)->sample(number);
    for (unsigned int j=1; j < batch.size(); ++j){
        samples.insert_rows(j,batch[j]->sample(number));
    }

    return samples;
}

frowvec NoiseBatch::eval(fmat* x){
	unsigned int dimensions = x->n_rows;
	unsigned int number = x->n_cols;
	frowvec result = zeros<frowvec>(x->n_cols);
	frowvec temp;

    for (unsigned int i=0; i< dimensions; ++i){
        temp = batch.at(i)->eval(x->row(i));
		for (unsigned int j = 0; j < number; ++j)
		{
			result(j) += temp(j);
		}
    }

    return result/(float)dimensions;
}

frowvec NoiseBatch::eval(float* x, unsigned int number){
	unsigned int numberOfDims = (unsigned int)batch.size();
    frowvec result = zeros<frowvec>(number);
	frowvec temp;

    for (unsigned int i=0; i< numberOfDims; ++i){
		temp = batch.at(i)->eval(x,number,i,numberOfDims);
		for (unsigned int j = 0; j < number; ++j)
		{
			result(j) += temp(j);
		}
    }

    return result/(float)numberOfDims;
}

void NoiseBatch::addNoise(Noise* newNoise){
    batch.push_back(newNoise);
}

NoiseBatch::NoiseBatch(){
}

NoiseBatch::~NoiseBatch(){

}
