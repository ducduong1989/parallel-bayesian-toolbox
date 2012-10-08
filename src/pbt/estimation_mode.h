#ifndef MODE_ESTIMATION
#define MODE_ESTIMATION

#include "estimation.h"

class EstimationMode: public Estimation {
private:
        int dim;
public:
        fmat getEstimation(Particles*);

        EstimationMode();
        ~EstimationMode();
};

#endif 
