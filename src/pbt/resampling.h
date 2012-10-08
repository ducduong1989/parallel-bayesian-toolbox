#ifndef RESAMPLING
#define RESAMPLING

#include "particles.h"
#include <vector>

//template <typename T, int DIM>
class Resampling{
protected:
    std::vector<int> assignmentVec;

public:
    virtual Particles resample(Particles*);
    std::vector<int> getAssignmentVec();
    
    ~Resampling();
    Resampling();
};

#endif 
