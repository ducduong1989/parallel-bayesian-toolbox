//particle.h
#ifndef PARTICLES
#define PARTICLES


#include <armadillo>
#include <vector>

using namespace arma;

//template <typename T, int DIM>
class Particles{

	public:
                fmat samples;     /*< Matrix of particles. Each particle equals one column of the matrix */
                frowvec weights;    /*< Vector of weights */
                int dim;        /*< Dimension of one particle */

                Particles();
                Particles(int);

                ~Particles();

                Particles(const Particles&);
                Particles &operator=(const Particles&);
};

#endif
