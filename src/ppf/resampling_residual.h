#ifndef RESIDUAL_RESAMPLING
#define RESIDUAL_RESAMPLING

#include "resampling.h"

//template <typename T, int DIM>
class ResamplingResidual : public Resampling{
private:

public:
    ResamplingResidual();
    ~ResamplingResidual();

    Particles resample(Particles*);
	
};

#endif 
