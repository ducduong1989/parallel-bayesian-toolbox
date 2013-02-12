#ifndef SIR_PF
#define SIR_PF

#include "bf.h"

/**
  * particle filter using Kullback-Leibler Divergence sampling
  */
class BFilterKLD : public BFilter {
private:
    unsigned int mxmin;
    unsigned int numBins;
    float eps;
    umat histogram;
    fvec minH;
    fvec maxH;

    bool testIfEmptyBin(fmat sample);
    void generateHistogram();
public:
    BFilterKLD();
    ~BFilterKLD();

    void setMxMin(unsigned int MxMin);
    void setEpsilon(float epsilon);
    void setNumberOfBins(unsigned int number);

    void predict();
    void update(fvec);
};

#endif
