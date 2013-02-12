#ifndef SIR_PF
#define SIR_PF

#include "bf.h"

/**
  * Sequential Importance Resampling Particle Filter
  */
class BFilterSIR : public BFilter {
public:
    BFilterSIR();
    ~BFilterSIR();

    void predict();
    void update(fvec);
};

#endif
