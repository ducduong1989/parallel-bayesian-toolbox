#ifndef RESAMPLING
#define RESAMPLING

#include "particles.h"
#include <vector>

//template <typename T, int DIM>
class Resampling{

public:
    virtual Particles resample(Particles*);
    
    ~Resampling();
    Resampling();
};

#endif 
