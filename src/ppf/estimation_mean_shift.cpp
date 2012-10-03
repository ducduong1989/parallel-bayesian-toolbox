#include "estimation_mean_shift.h"
#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14
#endif

fmat EstimationMeanShift::getEstimation(Particles *input){
    unsigned int particlesDimension = input->samples.n_rows;
    unsigned int numberOfParticles = input->samples.n_cols;
    unsigned int closestCenter;
    fvec estimation(particlesDimension);
    particles = input;
    float distanceToParticle;
    float weightSum;
    float weight;
    fvec diffVector;
    fvec currentCenter;
    fvec oldCenter;

#ifdef VERBOSE
    printf("calculating estimation of %d particles in %d dimensions\n",input->samples.n_cols,input->samples.n_rows);
    printf("Number of Classes: %i; maximal %i iterations\n",numClasses,maxIters);
#endif

    // initialize center fields
    centers = input->samples;

    for (unsigned int j=0; j< numberOfParticles; j++)
    {
        currentCenter = particles->samples.col(j);
        oldCenter = currentCenter;
        for (unsigned int i=0; i< maxIters;++i)
        {
            weightSum = 0;
            for (unsigned int k=0; k< numberOfParticles;++k){
                if (k != j)
                {
                    // calculate euclidian distance between particle and center
                    diffVector = centers.col(j)-particles->samples.col(k);
                    distanceToParticle = norm(diffVector,2);
                    if (distanceToParticle < dist)
                    {
                        // gaussian kernel function
                        weight = (1.0f/(2*(float)M_PI*dist))*exp(-0.5f*(distanceToParticle/dist)*(distanceToParticle/dist));
                        currentCenter += particles->samples.col(k)*weight;
                        weightSum += weight;
                    }
                }
            }
            // calculate new center position
            centers.col(j) = currentCenter/weightSum;

            // check is the center has moved or converged
            if (norm(currentCenter-oldCenter, 2) < 1)
            {
                break;
            }
            oldCenter = currentCenter;
        }
    }

    // check centers for duplicates
    // to do
    /*for (unsigned int j=0; j< numberOfParticles; j++)
    {

    }*/

    // check for best cluster and save center to estimation
    closestCenter = getClosestExistingCenter();
    for (unsigned int i = 0; i< particlesDimension; ++i){
        estimation(i) = centers(i,closestCenter);
    }

#ifdef VERBOSE
    printf("state estimated\n");
    printf("resulting number of classes: %i\n",numClasses);
#endif

    return estimation;
}

void EstimationMeanShift::setConfiguration(float distance, float epsilon, unsigned int maximumIterations){
    dist = distance;
    convergenceDistance = epsilon;
    maxIters = maximumIterations;
}

EstimationMeanShift::~EstimationMeanShift(){

}

EstimationMeanShift::EstimationMeanShift()
{
    convergenceDistance = 1.0f;
    maxIters = 1;
}

unsigned int EstimationMeanShift::getClosestExistingCenter()
{
    fvec differenceVector = reference - centers.col(0);
    float        smallestdistance = (float)norm(differenceVector,2);
    unsigned int closestcenter    = 0;

    for(unsigned int i = 1; i < numClasses; i++)
    {
        differenceVector = reference - centers.col(i);
        float   newdist = (float)norm(differenceVector,2);
        if( newdist < smallestdistance){
            smallestdistance = newdist;
            closestcenter = i;
        }
    }

    return closestcenter;
}

void EstimationMeanShift::setRefereceVector(fvec newReference)
{
    reference = newReference;
}
