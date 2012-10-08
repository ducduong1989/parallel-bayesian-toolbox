#ifndef SYSTEMATIC_RESAMPLING
#define SYSTEMATIC_RESAMPLING

#include "resampling.h"

//template <typename T, int DIM>
class ResamplingSystematic : public Resampling{
private:

public:
    ResamplingSystematic();
    ~ResamplingSystematic();

    Particles resample(Particles*);

};

#endif 
