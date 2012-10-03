#ifndef NOISE
#define NOISE

#include <armadillo>

using namespace arma;

class Noise {

public:

    float a;
    float b;
    float c;

    Noise();
    ~Noise();

    Noise(const Noise&);
    Noise &operator=(const Noise&);

    virtual frowvec sample(unsigned int);
    virtual frowvec eval(frowvec);
	virtual frowvec eval(float* input, int number, int dim, int numberOfDims);
};

#endif
