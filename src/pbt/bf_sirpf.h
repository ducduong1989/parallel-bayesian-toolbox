#ifndef SIR_PF
#define SIR_PF

#include "bf.h"

class BFilterSIR : public BFilter {
public:
    BFilterSIR();
    ~BFilterSIR();

    void predict();
    void update(fvec);
};

#endif
