#include "noise_uni.h"

frowvec NoiseUniform::sample(unsigned int number){
    frowvec samples;

    samples = a + randu<frowvec>(number)*(b-a);

    return samples;
}

frowvec NoiseUniform::eval(frowvec input){
    frowvec result = zeros<frowvec>(input.n_cols);

    for (unsigned int i = 0; i<input.n_cols; ++i)
    {
        if (( a <= input(i)) && (input(i) <= b))
        {
            result(i) = 1/(b-a);
        }
    }

    return result;
}

NoiseUniform::NoiseUniform(float leftBorder,float rightBorder){
    a = leftBorder;
    b= rightBorder;
}
