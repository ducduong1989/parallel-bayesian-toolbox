#ifndef MULTINOMIAL_RESAMPLING
#define MULTINOMIAL_RESAMPLING

#include "resampling.h"

//template <typename T, int DIM>
class ResamplingMultinomial : public Resampling{
private:

public:
    ResamplingMultinomial();
    ~ResamplingMultinomial();

    Particles resample(Particles*);

};

#endif 
