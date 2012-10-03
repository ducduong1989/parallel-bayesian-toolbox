#include "pf_sir_cuda.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

PFilterSIRCUDA::PFilterSIRCUDA(){

}

void PFilterSIRCUDA::predict(){

    samplesOnGPU = process->ffun_gpu(&particles.samples);

    normalizeWeights();
}

void PFilterSIRCUDA::update(fvec measurement){
    float neff=0.0;
    float* measurementDev;
    cudaMalloc( &measurementDev, (size_t) measurement.n_rows * sizeof(float)) ;
    cudaMemcpy(measurementDev,measurement.memptr(),(size_t) measurement.n_rows * sizeof(float), cudaMemcpyHostToDevice);

    //fmat virtualMeasurementOfParticles;
    //fmat differences = zeros<fmat>(measurement.n_rows,particles.samples.n_cols);
    frowvec evals;

    //virtualMeasurementOfParticles = process->hfun(&particles.samples);
    measurementOnGPU = process->hfun_gpu(samplesOnGPU, particles.samples.n_cols, particles.samples.n_rows);

	// calculate differences
    //differences = virtualMeasurementOfParticles - inputMatrix;
    /*for (unsigned int i=0; i< virtualMeasurementOfParticles.n_cols; ++i)
	{
		for(unsigned int j=0; j<virtualMeasurementOfParticles.n_rows;++j)
		{
			differences(j,i) = virtualMeasurementOfParticles(j,i) - measurement(j);
		}
    }*/

    callDeviationKernel(measurementOnGPU, measurementDev, particles.samples.n_rows,
                        particles.samples.n_cols, deviationsOnGPU);

    evals = process->eval_gpu(deviationsOnGPU, particles.samples.n_cols);

	// % is the Schur product (elementwise vector multiplication
    particles.weights = particles.weights % evals;

    // get samples from graphics card
    cudaMemcpy(particles.samples.memptr(),samplesOnGPU, particles.samples.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

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


PFilterSIRCUDA::~PFilterSIRCUDA(){
}
