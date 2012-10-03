#include "noise_tri.h"

frowvec NoiseTriangular::sample(unsigned int number){
    frowvec samples;
    frowvec r1 = randu<frowvec>(number);
    frowvec r2 = randu<frowvec>(number);

    samples = a + ((b-a)/2) * (r1+r2) ;

    return samples;
}

frowvec NoiseTriangular::eval(frowvec input){
    frowvec result;
    float x = (a+b)/2;

    fmat temp = 1 -( (2 * abs(input-x))/(b-a));
    temp.insert_rows(1,zeros<frowvec>(input.n_cols));

    result = (2/(b-a))*max(temp,0);

    return result;
}

NoiseTriangular::NoiseTriangular(float leftBorder, float rightBorder){
    a=leftBorder;
    b=rightBorder;
}

NoiseTriangular::~NoiseTriangular(){

}

