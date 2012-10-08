#include "estimation_kmeans.h"
#include <vector>
#include <iostream>

fmat EstimationKMeans::getEstimation(Particles *input){
    unsigned int particlesDimension = input->samples.n_rows;
    unsigned int numberOfParticles = input->samples.n_cols;
    fvec estimation(particlesDimension);
    unsigned int newAssign = 0;
    particles = input;
    bool somethingChanged = false;
    unsigned int closestCenter;

#ifdef VERBOSE
    printf("calculating estimation of %d particles in %d dimensions\n",input->samples.n_cols,input->samples.n_rows);
    printf("Number of Classes: %i; maximal %i iterations\n",numClasses,maxIters);
#endif

    // initialize assign vector
    assign = zeros<ucolvec>(numberOfParticles);
    // initialize center fields
    centers = zeros<fmat>(particlesDimension, numClasses);
    centerCounts = zeros<ucolvec>(numClasses);
    centerWeights = zeros<frowvec>(numClasses);

    initCenters();

    for (unsigned int i=0; i< maxIters;++i)
    {
        somethingChanged = false;

        // calculate assignments
        for (unsigned int j=0; j< particles->samples.n_cols; j++)
        {
            newAssign = getClosestCenter(j);
            if (newAssign != assign(j))
            {
                somethingChanged = true;
                assign(j) = newAssign;
                centerCounts(newAssign) ++;
                centerWeights(newAssign) = particles->weights(j);
            }
        }

        if (somethingChanged)
        {
            // calculate new centers
            for (unsigned int j=0; j< particles->samples.n_cols; j++)
            {
                for (unsigned int k=0; k< particlesDimension;++k)
                {
                    centers(k,assign(j)) += particles->samples(k,j) * particles->weights(j);
                }
            }
            for (unsigned int g=0; g<numClasses; ++g){
                if (centerCounts(g))
                {
                    for (unsigned int k=0; k< particlesDimension;++k)
                    {
                        centers(k,g) = centers(k,g)/centerWeights(g);
                    }
                }
            }
        }
        else
        {
            break;
        }

    }

    // check for best cluster and save center to estimation
    closestCenter = getClosestExistingCenter();
    for (unsigned int i = 0; i< particlesDimension; ++i){
        estimation(i) = centers(i,closestCenter);
    }

#ifdef VERBOSE
    printf("state estimated");
#endif

    return estimation;
}

void EstimationKMeans::setConfiguration(unsigned int numberOfClasses, unsigned int maximumIterations){
    numClasses = numberOfClasses;
    maxIters = maximumIterations;
}

EstimationKMeans::~EstimationKMeans(){

}

EstimationKMeans::EstimationKMeans()
{
    numClasses = 1;
    maxIters = 1;
}

unsigned int EstimationKMeans::getClosestCenter(unsigned int particle)
{
    fvec differenceVector = particles->samples.col(particle)-centers.col(0);
    float        smallestdistance = (float)norm(differenceVector,2);
    unsigned int closestcenter    = 0;

    for(unsigned int i = 1; i < numClasses; i++)
    {
        differenceVector = particles->samples.col(particle)-centers.col(i);
        float   newdist = (float)norm(differenceVector,2);
        if( newdist < smallestdistance){
            smallestdistance = newdist;
            closestcenter = i;
        }
    }

    return closestcenter;
}

unsigned int EstimationKMeans::getClosestExistingCenter()
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

void EstimationKMeans::initCenters()
{
    // take the first particles as first guess of cluster positions
    // maybe biased but no need to generate random numbers
    for(unsigned int j = 0; j < numClasses; j++)
    {
        for (unsigned int i=0; i< centers.n_rows; ++i)
        {
            centers(i,j) = particles->samples(i,j);
        }
    }
}

void EstimationKMeans::setRefereceVector(fvec newReference)
{
    reference = newReference;
}
