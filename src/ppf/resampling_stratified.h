#ifndef RESAMPLING_STRATIFIED
#define RESAMPLING_STRATIFIED

#include "resampling.h"

//template<typename T, int DIM>
class ResamplingStratified : public Resampling{
private:

public:
    ResamplingStratified();
    ~ResamplingStratified();

    Particles resample(Particles*);
};

#endif 
