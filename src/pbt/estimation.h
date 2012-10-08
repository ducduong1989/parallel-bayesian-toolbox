#ifndef ESTIMATION
#define ESTIMATION

#include "particles.h"
#include <vector>

//template<typename T, int DIM>
class Estimation{

public:
        virtual fmat getEstimation(Particles*);

        ~Estimation();
        Estimation();


};

#endif
