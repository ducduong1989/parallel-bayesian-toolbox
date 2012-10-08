#include "bf.h"
#include <stdio.h>
#include <iostream>

BFilter::BFilter()
{
    nthr = 5;
}

BFilter::~BFilter()
{

}

/**
  * Reads the particles object
  */
Particles BFilter::getParticles()
{
    return particles;
}

fmat BFilter::getCovariance()
{
    unsigned int numParticles = particles.samples.n_cols;
    unsigned int dim = particles.samples.n_rows;
    fmat Sk = zeros<fmat>(dim, dim);
    fvec mean = zeros<fvec>(dim);
    float preterm = 0;

    float sumOfSquaredWeights = 0;
    float sumOfWeights = 0;

    for (unsigned int j=0; j < numParticles; ++j)
    {
        sumOfWeights += particles.weights(j);
        sumOfSquaredWeights += particles.weights(j)*particles.weights(j);

        for (unsigned int i = 0; i< dim; ++i)
        {
            mean(i) += particles.weights(j) * particles.samples(i,j);
        }
    }
    mean = mean / sumOfWeights;

    preterm = 1 / (1 - sumOfSquaredWeights);

    for (unsigned int j=0; j < dim; ++j)
    {
        for (unsigned int i = 0; i < dim; ++i)
        {
            for (unsigned int k=0; k < numParticles; ++k)
            {
                float diff1 = particles.samples(i,k) - mean(i);
                float diff2 = particles.samples(j,k) - mean(j);

                Sk(i,j) += particles.weights(k) * diff1 *diff2 ;
            }
            Sk (i,j) *= preterm;
        }
    }
	return Sk;
}

/**
  * Manipulate the particles including samples and their weights
  */
void BFilter::setParticles(fmat newSamples,frowvec newWeights)
{
    nthr = (unsigned int)floor(((float)nthr/number)*newSamples.n_cols);
    number = newSamples.n_cols;
    if (number != newWeights.n_cols){
        perror("number of particles and weights is inconsistent!");
        return;
    }
    setSamples(newSamples);
    setWeights(newWeights);
}

/**
  * Manipulate only the samples of the partilces object
  */
void BFilter::setSamples(fmat newSamples)
{
    particles.samples = newSamples;
}

/**
  * Manipulate only the weights of the partilces object
  */
void BFilter::setWeights(frowvec newWeights)
{
    particles.weights = newWeights;
}

/**
  * pressing prediction step including a temporary estimation
  */
void BFilter::predict()
{
    // step forward
    particles.samples = process->ffun(&particles.samples);

    normalizeWeights();
}

fvec BFilter::getEstimation(){
	return estimator->getEstimation(&particles);

}

/**
  * calculate weights so that sum of all weights equals 1
  */
void BFilter::normalizeWeights()
{
    // normalize weights so that sum of all weights equals 1
    float sum=0.0;
    for (unsigned int i=0; i < particles.weights.n_cols;++i){
        sum = sum + particles.weights(i);
    }
    particles.weights = particles.weights / sum;
}

/**
  * calculate \f$ Neff = \frac{1}{sum(weight^2)} \f$
  */
float BFilter::calculateNeff()
{
    float neff=0;

    for (unsigned int i=0; i < particles.weights.n_cols;++i){
        neff = neff + (particles.weights(i)*particles.weights(i));
    }
    if (neff > 0){
        neff = (float)1.0 / neff;
    }
    else {
        neff = 0;
    }


    return neff;
}

/**
  * processing update step including resampling if possible
  */
void BFilter::update(fvec measurement)
{

    float neff=0.0;
    fmat virtualMeasurementOfParticles;
    fmat differences = zeros<fmat>(particles.samples.n_rows,particles.samples.n_cols);
    fvec evals;

    // generate simulated measurements
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

    // evaluation of particles
    evals = process->eval(&differences);
    // calculating new particle weights
    particles.weights = particles.weights * evals;

    normalizeWeights();

    neff=calculateNeff();
    if (neff <= nthr){
        // Neff is too low so all particles are weighted with standard values
#ifdef VERBOSE
        printf("too few particles. 1/N for all particles\n");
#endif
        standardWeighting();
    }
    else {
        // Neff is high enough to resample particles
        particles = resampler->resample(&particles);
    }
}

/**
  * Weights all particles with the same value 1/numberOfParticles
  */
void  BFilter::standardWeighting()
{
    for (unsigned int i=0 ; i<particles.weights.n_cols; i++){
            particles.weights(i) = (float)1.0 / (float)particles.weights.n_cols;
    }
}

/**
  * Set model used for particle generation
  */
void BFilter::setModel(Model *newModel)
{

    process = newModel;
    process->initialize();
}

/**
  * Set estimation method
  */
void BFilter::setEstimation(Estimation *newEstimator)
{

    estimator = newEstimator;
}

/**
  * Set resampling method
  */
void BFilter::setResampling(Resampling *newResampler)
{
    resampler = newResampler;
}

/**
  * Set number of particles. If the number of particles is reset all particles represent a zero vector and
  have standard weights.
  */
/*void PFilter::setNumberOfParticles(int numberOfParticles){
	nthr = (unsigned int)floor(((float)nthr/number) * numberOfParticles);
    number = numberOfParticles;

    particles.samples.set_size(dim,numberOfParticles);
    particles.samples.zeros();

    particles.weights = zeros<fmat>(1,numberOfParticles);

    standardWeighting();
}*/

/**
  * set threshold from 0 to 1
  */
void BFilter::setThresholdByFactor(float newThreshold)
{
    nthr = (unsigned int)floor(newThreshold * particles.weights.n_cols);
}

/**
  * set threshold as a defined number
  */
void BFilter::setThresholdByNumber(unsigned int newThreshold)
{
    nthr = newThreshold;
}
