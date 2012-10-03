#ifndef SIR_PF_CUDA
#define SIR_PF_CUDA

#include "pf_cuda.h"

class PFilterSIRCUDA : public PFilterCUDA
{
public:
    PFilterSIRCUDA();
    ~PFilterSIRCUDA();

    void predict();
    void update(fvec);
};

#endif
