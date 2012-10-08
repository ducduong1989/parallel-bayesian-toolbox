#include "bf_llpf.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

BFilterLLPF::BFilterLLPF()
{

}

float BFilterLLPF::evalParticle(float input, float dev){
    float result;

    float preterm = (float) (1.0f/(dev*sqrt(2.0f*M_PI)));

    result = preterm * exp(-0.5f*((input/dev)*(input/dev)));

    return result;
}

void BFilterLLPF::predict()
{
    if (particles.samples.n_cols != covariances.size())
    {
        covariances.clear();
        for(int i = 0; i < particles.samples.n_cols; ++i)
        {
            covariances.push_back(eye<fmat>(process->getProcessDimension(),process->getProcessDimension()));
        }
    }

    //particles.samples = process->ffun(&particles.samples);
    for(int i = 0; i < particles.samples.n_cols; ++i)
    {
        fmat tempSample = (fmat)particles.samples.col(i);
        particles.samples.col(i) = process->ffun(&tempSample);

        // calculate covariance matrix using the Jacobian F and process noise
        covariances.at(i) = process->F*covariances.at(i)*process->F.t();
        covariances.at(i) += process->W;
    }
}

void BFilterLLPF::update(fvec measurement){
    float neff=0.0;
    fmat virtualMeasurementOfParticles;
    fmat differences = zeros<fmat>(measurement.n_rows,particles.samples.n_cols);
    frowvec evals;
    frowvec evalNormalization = ones<frowvec>(particles.samples.n_cols);
    fmat tempCovariance;
    std::vector<fmat> resampledCovariances;
    std::vector<int> assignmentVec;

    for(int i = 0; i < particles.samples.n_cols; ++i)
    {
        fvec vk;
        fmat Sk;
        fmat K;
        fmat I;

        // calculating Kalman gain using measurement Jacobian H and variance matrix
        Sk = process->H * covariances.at(i) * process->H.t() + process->W;
        K = covariances.at(i) * process->H.t() * Sk.i();

        // difference between real and predicted measurement
        vk = measurement - process->hfun(&particles.samples);

        // calculating updated mean and coviriance
        particles.samples = particles.samples + K*vk;
        I.eye(particles.samples.n_rows, particles.samples.n_rows);
        covariances.at(i) = (I - K * process->H) * covariances.at(i);
    }

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
    // % is the Schur product (elementwise vector multiplication
    particles.weights = particles.weights % evals;

    // calculate normalization depending on the calculated covariance
    for(int i = 0; i < particles.samples.n_cols; ++i)
    {
        tempCovariance = covariances.at(i);
        for (int j = 0; j < particles.samples.n_rows; ++j )
        {
            evalNormalization(i) *= evalParticle(differences(j,i),tempCovariance(j,j));
        }
    }

    particles.weights = particles.weights / evalNormalization;

    normalizeWeights();

    neff=calculateNeff();

    if (neff <= nthr){
#ifdef VERBOSE
        printf("too few particles. 1/N for all particles\n");
#endif
        particles.weights = ones<frowvec>(particles.weights.n_cols) / (float)particles.weights.n_cols;
    }
    else    particles = resampler->resample(&particles);

    // assing resampled set of covariances
    assignmentVec = resampler->getAssignmentVec();
    for (int i = 0; i < assignmentVec.size(); ++i)
    {
        resampledCovariances.push_back(covariances.at(assignmentVec.at(i)));
    }
    covariances = resampledCovariances;
}



BFilterLLPF::~BFilterLLPF()
{
}
