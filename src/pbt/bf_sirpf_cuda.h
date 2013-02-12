#ifndef SIR_PF_CUDA
#define SIR_PF_CUDA

#include "bf_cuda.h"

/**
  * GPU accelerated Sequential Importance Resampling Particle Filter
  */
class BFilterSIRCUDA : public BFilterCUDA
{
public:
    BFilterSIRCUDA();
    ~BFilterSIRCUDA();

    void predict();
    void update(fvec);
};

#endif
