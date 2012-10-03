#ifndef SIR_PF
#define SIR_PF

#include "pf.h"

class BFilterSIR : public PFilter {
public:
    BFilterSIR();
    ~BFilterSIR();

    void predict();
    void update(fvec);
};

#endif
