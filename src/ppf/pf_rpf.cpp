#include "pf_rpf.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

BFilterRPF::BFilterRPF()
{

}

void BFilterRPF::predict()
{

    particles.samples = process->ffun(&particles.samples);

    normalizeWeights();
}


void BFilterRPF::update(fvec measurement)
{
    unsigned int dim = particles.samples.n_rows;
    float h_opt = pow((float)(4.0f/(dim+2.0f)),(float)(1.0f/(dim+4.0f)))*pow((float)particles.samples.n_cols,(-1.0f/(dim+4.0f)));
    fmat Dk;
    fmat Sk; // empirical covariance matrix of weighted samples
    float neff=0.0;
    fmat virtualMeasurementOfParticles;
    fmat differences = zeros<fmat>(measurement.n_rows,particles.samples.n_cols);
    frowvec evals;

    virtualMeasurementOfParticles = process->hfun(&particles.samples);

    // calculate differences
    //differences = virtualMeasurementOfParticles - inputMatrix;
	for (unsigned int i=0; i< virtualMeasurementOfParticles.n_cols; ++i)
	{
		for(unsigned int j=0; j<virtualMeasurementOfParticles.n_rows;++j)
		{
			differences(j,i) = virtualMeasurementOfParticles(j,i) - measurement(j);
		}
	}

    evals = process->eval(&differences);

    particles.weights = particles.weights % evals;

    normalizeWeights();

    neff=calculateNeff();

    Sk = getCovariance();
    Dk = Sk%Sk;
    Dk = sqrt(Dk);
    Dk = sqrt(Dk);
    if (neff <= nthr){
#ifdef VERBOSE
        printf("too few particles. 1/N for all particles\n");
#endif
        particles.weights = ones<frowvec>(particles.weights.n_cols) / (float)particles.weights.n_cols;
    }
    else    particles = resampler->resample(&particles);

    particles.samples = particles.samples + h_opt*Dk*randn<fmat>(dim,particles.samples.n_cols);
}

BFilterRPF::~BFilterRPF(){
}
