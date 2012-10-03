#ifndef NOISE_BATCH
#define NOISE_BATCH

#include "noise.h"
#include <vector>
#include <armadillo>

using namespace arma;

class NoiseBatch {
public:
        std::vector<Noise*> batch;

		fmat sample(unsigned int);
		frowvec eval(fmat*);

		frowvec eval(float* x, unsigned int number);

        void addNoise(Noise*);

        NoiseBatch();
        ~NoiseBatch();
};

#endif
