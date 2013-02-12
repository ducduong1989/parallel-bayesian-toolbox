#ifndef RPF_PF
#define RPF_PF

#include "bf.h"

/**
  * Regularized Particle Filter
  */
class BFilterRPF : public BFilter {
public:
    BFilterRPF();
    ~BFilterRPF();

    void predict();
    void update(fvec);
};

#endif
