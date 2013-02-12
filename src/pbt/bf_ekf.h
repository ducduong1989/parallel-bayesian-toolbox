#ifndef EKF_PF
#define EKF_PF

#include "bf.h"

/**
  * Extended Kalman Filter
  */
class BFilterEKF : public BFilter {
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
