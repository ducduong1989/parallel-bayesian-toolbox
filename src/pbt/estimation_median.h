#ifndef MEDIAN_ESTIMATION
#define MEDIAN_ESTIMATION

#include "estimation.h"

//template<typename T, int DIM>
class EstimationMedian: public Estimation {
private:
        int dim;
public:
        fmat getEstimation(Particles*);

        EstimationMedian();
        ~EstimationMedian();
};

#endif
