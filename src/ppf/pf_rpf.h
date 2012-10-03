#ifndef RPF_PF
#define RPF_PF

#include "pf.h"

class BFilterRPF : public PFilter {
public:
    BFilterRPF();
    ~BFilterRPF();

    void predict();
    void update(fvec);
};

#endif
