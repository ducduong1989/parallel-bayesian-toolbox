// particlefilter.h
#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER

#include <vector>
#include "estimation.h"
#include "resampling.h"
#include "model.h"

//template <typename T, int DIM>
class BFilter
{
protected:
        unsigned int number;

        unsigned int nthr;

        fmat inputMatrix;

        Particles particles;
        Resampling *resampler;
        Estimation *estimator;
        Model *process;

        void standardWeighting();

        void normalizeWeights();

        float calculateNeff();

public:
        BFilter();

        ~BFilter();

        /**
          * methods to manipulate the particles set, e.g. number of particles
          * or weights
          */


        /**
          * sets the resampling threshold as a defined number of particles
          */
        void setThresholdByNumber(unsigned int);

        /**
          * sets the threshold for resampling as a factor from 0 to 1. It is
          * multiplied with the number of existing particles
          */
        void setThresholdByFactor(float);

        /**
          * gives the current set of particles including samples and weigts
          */
        Particles getParticles();
        void setParticles(fmat,frowvec);
        void setSamples(fmat);
        void setWeights(frowvec);

        fvec getEstimation();
        fmat getCovariance();

        void setInput(fvec);

        virtual void predict();
        virtual void update(fvec);
	
        void setEstimation(Estimation*);

        void setResampling(Resampling*);

        void setModel(Model*);
};

#endif 
