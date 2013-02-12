#ifndef ESTIMATION
#define ESTIMATION

#include "particles.h"
#include <vector>

/**
  * base class for estimation methods
  */
class Estimation{

public:
        virtual fmat getEstimation(Particles*);

        ~Estimation();
        Estimation();


};

#endif
