#ifndef MULTINOMIAL_NAIVE_DELETE
#define MULTINOMIAL_NAIVE_DELETE

#include "resampling.h"

//template <typename T, int DIM>
class ResamplingNaiveDelete : public Resampling{
private:
    int increaseThreshold;
    float decreaseThreshold;
public:
    ResamplingNaiveDelete();
    ~ResamplingNaiveDelete();

    Particles resample(Particles*);

    void setThresholds( float, unsigned int);

};

#endif 
