#ifndef EKF_PF
#define EKF_PF

#include "pf.h"

class BFilterEKF : public PFilter {
private:
    fmat P; // covariance matrix
public:
    BFilterEKF();
    ~BFilterEKF();

    void predict();
    void update(fvec);
    fmat getCovariance();
};

#endif
