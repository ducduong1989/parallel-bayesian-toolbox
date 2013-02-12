#ifndef LLPF_PF
#define LLPF_PF

#include "bf.h"
#include "bf_ekf.h"
#include <vector>

/**
  * Local Linearization Particle Filter
  */
class BFilterLLPF : public BFilter {
private:
    std::vector<fmat> covariances;

    float evalParticle(float input, float dev);

public:
    BFilterLLPF();
    ~BFilterLLPF();

    void predict();
    void update(fvec);
};

#endif
