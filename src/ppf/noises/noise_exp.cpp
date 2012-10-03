#include "noise_exp.h"

frowvec NoiseExponential::sample(unsigned int number){
    frowvec samples;

    samples = -a * log(randu<frowvec>(number));

    return samples;
}

frowvec NoiseExponential::eval(frowvec input){
    frowvec result = zeros<frowvec>(input.n_cols);

    for (unsigned int i = 0; i< input.n_cols; ++i)
    {
        if (input(i)>0)
        {
            result(i) = exp(-input(i)/a)/a;
        }
        // no else for zero needed because result was zero vector at start
    }


    return result;
}

NoiseExponential::NoiseExponential(float x){
    a = x;
}


NoiseExponential::~NoiseExponential(){

}
