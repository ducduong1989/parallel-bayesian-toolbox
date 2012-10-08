#include "bf_kld.h"
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "estimation_mean.h"

BFilterKLD::BFilterKLD()
{
    numBins = 150;
    eps = 0.1f;
    mxmin = 1800;
}

void BFilterKLD::predict()
{

}

void BFilterKLD::update(fvec measurement){
    unsigned int numberOfPredefinedRandomNumbers = particles.samples.n_cols;
    Particles newSet;
    newSet.samples = fmat(particles.samples.n_rows,0);
    newSet.weights = fmat(1,0);
    fmat newSample;
    fmat measurementDiff;
    fmat newWeights;
    float z095 = 1.6f;
    unsigned int m=0;
    unsigned int randomIndex = 0;
    float mx=(float)mxmin;
    float tempTerm = 0;
    float longTempTerm = 0;
    unsigned int k=0;
    unsigned int i=0;
    frowvec cumWeights = cumsum(particles.weights);
    //cumWeights.print();
    //float tempCumSum = cumWeights(cumWeights.n_cols-1);
    //cumWeights = cumWeights / tempCumSum;
    // pregenerate random numbers as vector due to performance reasons depending on the number of particles
    fvec randomNumbers = randu<fvec>(numberOfPredefinedRandomNumbers);
    float currentRandomNumber = 0;

    generateHistogram();

    while ((m<mx) || (m<mxmin))
    {
        i=0;
        // set the random number to look for and check the cummulative sum vector
        currentRandomNumber = randomNumbers(randomIndex);
        for (unsigned int r=0; r<cumWeights.n_cols-1; ++r)
        {
            if ((cumWeights(r) <= currentRandomNumber) &&
                (cumWeights(r+1) > currentRandomNumber))
            {
                i=r;
                break;
            }
        }
        //fmat oldsample = particles.samples.col(i);
        //printf("Old\n");oldsample.print();
        fmat oldSample = fmat(particles.samples.col(i));
        newSample = process->ffun(&oldSample);
        newSet.samples.insert_cols(m,newSample);
        //printf("New\n");newSample.print();
        //printf("Meas\n");measurement.print();

        measurementDiff = process->hfun(&newSample)-measurement;
        //printf("Diff\n");measurementDiff.print();
        newWeights = process->eval(&measurementDiff);
        //printf("Weight\n");newWeights.print();
        newSet.weights.insert_cols(m,newWeights);

        //test if new sample falls in empty bin of histogram
        if (testIfEmptyBin(newSample))
        {
            k++;
            if (k>1)
            {
                tempTerm = 2.0f/(9.0f*(k-1));
                longTempTerm = 1.0f-tempTerm+sqrt(tempTerm)*z095;
                mx=((k-1)/(2*eps))*longTempTerm*longTempTerm*longTempTerm;
            }
        }

        m++;
        randomIndex++;
        if (randomIndex >= numberOfPredefinedRandomNumbers)
        {
            randomNumbers = randu<fvec>(numberOfPredefinedRandomNumbers);
            randomIndex = 0;
        }
    }
    particles = newSet;
    normalizeWeights();
}

void BFilterKLD::setMxMin(unsigned int MxMin)
{
    mxmin = MxMin;
}

void BFilterKLD::setEpsilon(float epsilon)
{
    eps = epsilon;
}

void BFilterKLD::setNumberOfBins(unsigned int number)
{
    numBins = number;
}

void BFilterKLD::generateHistogram()
{
    float squaredWeightSum;
    float sumOfSquaredDeviations = 0;
    frowvec squaredWeights;
    EstimationMean estimation;
    fvec mean;
    fvec cov;
    fvec nextMean;
    float numOfSigma = 3;
    Particles nextStep;

    // generate a possible set of particles for next step
    nextStep.samples = process->ffun(&particles.samples);
    nextStep.weights = particles.weights;

    // calculate mean of current partilce set
    mean = estimation.getEstimation(&particles);
    cov = zeros<fvec>(mean.n_rows);
    // set all histogram bins to zero
    histogram = zeros<umat>(particles.samples.n_rows, numBins);

    // calculate diagonal of covariance matrix of current particle set
    //p_cov = (1 - w(1,:)*w(1,:)')^(-1) * p_cov; % weighted sample covariance
    squaredWeights = particles.weights % particles.weights;
    squaredWeightSum = sum(squaredWeights);
    for (unsigned int i=0; i< mean.n_rows; ++i)
    {
        sumOfSquaredDeviations = 0;
        for (unsigned j = 0; j< particles.samples.n_cols; ++j)
        {
            sumOfSquaredDeviations += particles.weights(j)*(particles.samples(i,j)-mean(i))*(particles.samples(i,j)-mean(i));
        }
        cov(i) = (1.0f / (1.0f - squaredWeightSum)) * sumOfSquaredDeviations;
    }

    // calculate mean of next particle set
    nextMean = estimation.getEstimation(&nextStep);

    // calculate min and max hof histograms
    minH = nextMean - numOfSigma * cov;
    maxH = nextMean + numOfSigma * cov;
}

bool BFilterKLD::testIfEmptyBin(fmat sample)
{
    float step;
    unsigned int index;
    bool oneEmpty = false;

    for (unsigned int i=0; i< sample.n_rows;++i)
    {
        if ((sample(i) >= minH(i)) && (sample(i) <= maxH(i)))
        {
            // calculate the index of sample in current histogram dimension
            step = (maxH(i) - minH(i))/numBins;
            index = (int)((sample(i,0) - minH(i)) / step);
            // check if bin is empty
            if (histogram(i,index) == 0)
            {
                // if sample falls into empty bin, set bin as full and return true
                histogram(i,index) = 1;
                oneEmpty = true;
                return oneEmpty;
            }
        }
    }

    return oneEmpty;
}

BFilterKLD::~BFilterKLD()
{
}
