#include "noise_trap.h"

frowvec NoiseTrapezoidal::sample(unsigned int number){
    frowvec samples;
    frowvec r1 = randu<frowvec>(number);
    frowvec r2 = randu<frowvec>(number);

    samples = a + ((b-a)/2) * ((1+beta)*r1 + (1-beta)*r2 );

    return samples;
}

frowvec NoiseTrapezoidal::eval(frowvec input){
    frowvec result;
    frowvec x = ones<frowvec>(input.n_cols)*((a+b)/2);

    result = 1/(lambda1+lambda2) * min (1/(lambda2-lambda1)*max(lambda2-abs(input-x),0),1);

    return result;
}

NoiseTrapezoidal::NoiseTrapezoidal(float newA1,float newA2,float newB1,float newB2){
    a1 = newA1;
    a2 = newA2;
    b1 = newB1;
    b2 = newB2;

    a = a1+a2;
    b = b1 + b2;

    lambda1 = abs((b1-a1)-(b2-a2))/2;
    lambda2 = (b-a)/2;

    beta = lambda1/lambda2;

}
