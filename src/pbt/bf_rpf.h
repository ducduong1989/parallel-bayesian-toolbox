#ifndef RPF_PF
#define RPF_PF

#include "bf.h"

class BFilterRPF : public BFilter {
public:
    BFilterRPF();
    ~BFilterRPF();

    void predict();
    void update(fvec);
};

#endif
