// particlefilter.h
#ifndef PARTICLE_FILTER
#define PARTICLE_FILTER

#include <vector>
#include "estimation.h"
#include "resampling.h"
#include "model.h"

/**
  * Bayes Filter base class implementing a simple SIS particle filter
  */
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

        void setThresholdByNumber(unsigned int);
        void setThresholdByFactor(float);

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
