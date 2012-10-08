#include "bf_ekf.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

BFilterEKF::BFilterEKF()
{

}

fmat BFilterEKF::getCovariance()
{
    return P;
}

void BFilterEKF::predict()
{

    particles.samples = process->ffun(&particles.samples);

    // calculate covariance matrix using the Jacobian F and process noise
    P = process->F*P*process->F.t();
    P += process->W;

}

void BFilterEKF::update(fvec measurement)
{
    fvec vk;
    fmat Sk;
    fmat K;
    fmat I;

    // calculating Kalman gain using measurement Jacobian H and variance matrix
    Sk = process->H * P * process->H.t() + process->W;
    K = P * process->H.t() * Sk.i();

    // difference between real and predicted measurement
    vk = measurement - process->hfun(&particles.samples);

    // calculating updated mean and coviriance
    particles.samples = particles.samples + K*vk;
    I.eye(particles.samples.n_rows, particles.samples.n_rows);
    P = (I - K * process->H) * P;

}


BFilterEKF::~BFilterEKF()
{
}
