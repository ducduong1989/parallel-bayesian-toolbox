#include "noise_gamma.h"


frowvec NoiseGamma::sample(unsigned int number){
    frowvec samples;

    fmat r = randu<fmat>(q, number);

    samples = log(prod(r,0));

    return samples;
}

frowvec NoiseGamma::eval(frowvec input){

    frowvec result = zeros<frowvec>(input.n_cols);

    for (unsigned int i = 0; i< input.n_cols; ++i)
    {
        if (input(i)>0)
        {
            result(i) = pow((float)input(i),(float)q)*exp(-input(i))/factorial(q);
        }
        // no else for zero needed because result was zero vector at start
    }


    return result;
}

long NoiseGamma::factorial(unsigned int n)
{
    if (n > 0){
        return n*factorial(n-1);
    }
    else
    {
        return 1;
    }
}

NoiseGamma::NoiseGamma(unsigned int order){
    q = order;
}

NoiseGamma::~NoiseGamma(){

}

