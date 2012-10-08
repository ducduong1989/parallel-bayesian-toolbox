#include "bf_sirpf.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

BFilterSIR::BFilterSIR(){

}

void BFilterSIR::predict(){

    particles.samples = process->ffun(&particles.samples);

    normalizeWeights();
}

void BFilterSIR::update(fvec measurement){
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
    evals.print();
	// % is the Schur product (elementwise vector multiplication
    particles.weights = particles.weights % evals;

    normalizeWeights();

    neff=calculateNeff();

    if (neff <= nthr){
#ifdef VERBOSE
        printf("too few particles. 1/N for all particles\n");
#endif
        particles.weights = ones<frowvec>(particles.weights.n_cols) / (float)particles.weights.n_cols;
    }
    else    particles = resampler->resample(&particles);
}


BFilterSIR::~BFilterSIR(){
}
