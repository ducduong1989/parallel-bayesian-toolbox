#ifndef UKF_PF
#define UKF_PF

#include "bf.h"

/**
  * Unscented Kalman Filter
  */
class BFilterUKF : public BFilter {
private:
    fvec x1;
    fmat X1;
    fmat P1;
    fmat X2;

    fvec z1;
    fmat Z1;
    fmat P2;
    fmat Z2;

    fmat Wm;
    fmat Wc;

    fmat P;

    fmat sigmas(fvec x, fmat P, float c);
    void utProcess(fmat X,fvec Wm, fvec Wc, unsigned int n, fmat R);
    void utMeasurement(fmat X, fvec Wm, fvec Wc, unsigned int n, fmat R);
public:
    BFilterUKF();
    ~BFilterUKF();

    void predict();
    void update(fvec);
    fmat getCovariance();
};

#endif
