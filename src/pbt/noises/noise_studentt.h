#ifndef __NOISE_STUDENT_T__
#define __NOISE_STUDENT_T__


#include "noise.h"
#include <cmath>

class NoiseStudentT : public Noise {
public:
    frowvec eval(frowvec);
    frowvec sample(unsigned int);

    NoiseStudentT(float,float);
    ~NoiseStudentT();
};

#endif
