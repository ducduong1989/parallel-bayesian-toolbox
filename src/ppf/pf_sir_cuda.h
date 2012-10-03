#ifndef SIR_PF_CUDA
#define SIR_PF_CUDA

#include "pf_cuda.h"

class BFilterSIRCUDA : public BFilterCUDA
{
public:
    BFilterSIRCUDA();
    ~BFilterSIRCUDA();

    void predict();
    void update(fvec);
};

#endif
