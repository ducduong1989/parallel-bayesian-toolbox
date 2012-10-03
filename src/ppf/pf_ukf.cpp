#include "pf_ukf.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

BFilterUKF::BFilterUKF(){

}

fmat BFilterUKF::getCovariance()
{
    return P;
}

void BFilterUKF::predict(){
    //L=numel(x);                                 //numer of states
    //m=numel(z);                                 //numer of measurements
    unsigned int numStates = particles.samples.n_cols;

    float alpha=1.0f;                                 //default 1e-3, tunable
    float kappa=0.0f;                                       //default, tunable
    float beta=2.0f;                                     //default, tunable

    //lambda=alpha^2*(L+kappa)-L;                    //scaling factor
    float lambda=alpha*alpha*(numStates+kappa)-numStates;   //scaling factor

    float c=numStates+lambda;                                 //scaling factor gamma
    //Wm=[lambda/c 0.5/c+zeros(1,2*L)];           //weights for means
    Wm=zeros<fmat>(1, 2*numStates+1) + (0.5f/c);             //weights for means
    Wm(0)=lambda/c;

    Wc=Wm;
    Wc(0)=Wc(0)+(1-alpha*alpha+beta);               //weights for covariance
    c=sqrt(c);
    fmat X=sigmas(fvec(particles.samples),P,c);                            //sigma points around x
    utProcess(X,Wm,Wc,numStates,process->Q);
}

void BFilterUKF::update(fvec measurement){
    //unscented transformation of process
    unsigned int numMeasurements = measurement.n_cols;
    utMeasurement(X1, Wm, Wc, numMeasurements, process->H);       //unscented transformation of measurments
    fmat P12=X2*Wc.diag()*Z2.t();                        //transformed cross-covariance
    fmat R = chol(P2);
    fmat K = (P12/R)/R.t(); // Filter gain.

    // K=P12*P2.i();
    particles.samples=x1+K*(measurement-z1);       //state update

    P=P1-K*P12.t();                                //covariance update


}

//function [y,Y,P,Y1]=ut(f,X,Wm,Wc,n,R)
void BFilterUKF::utProcess(fmat X,fvec Wm, fvec Wc, unsigned int n, fmat R)
{
    //Unscented Transformation
    //Input:
    //        f: nonlinear map
    //        X: sigma points
    //       Wm: weights for mean
    //       Wc: weights for covraiance
    //        n: numer of outputs of f
    //        R: additive covariance
    //Output:
    //        y: transformed mean
    //        Y: transformed smapling points
    //        P: transformed covariance
    //       Y1: transformed deviations

    unsigned int L=X.n_cols;
    x1 = zeros<fvec>(n);
    X1 = zeros<fmat>(n,L);
    //for k=1:L
    for (unsigned int k=0; k < L; ++k)
    {
        fmat XColK = X.col(k);
        X1.col(k)= process->ffun(&XColK);
        x1=x1+Wm(k)*X1.col(k);
    }

    //X2=X1-x1(:,ones(1,L));  // generate duplicates of vector x1
    X2 = X1;
    for (unsigned int j = 0; j < L; ++j)
    {
        for (unsigned int i = 0; i < x1.n_rows; ++i)
        {
            X2(i,j) -= x1(i);
        }
    }

    P1=X2*Wc.diag()*X2.t()+R;
}

//function [y,Y,P,Y1]=ut(f,X,Wm,Wc,n,R)
void BFilterUKF::utMeasurement(fmat X, fvec Wm, fvec Wc, unsigned int n, fmat R)
{
    //Unscented Transformation
    //Input:
    //        f: nonlinear map
    //        X: sigma points
    //       Wm: weights for mean
    //       Wc: weights for covraiance
    //        n: numer of outputs of f
    //        R: additive covariance
    //Output:
    //        y: transformed mean
    //        Y: transformed smapling points
    //        P: transformed covariance
    //       Y1: transformed deviations

    unsigned int L=X.n_cols;
    z1=zeros<fvec>(n);
    Z1=zeros<fmat>(n,L);
    //for k=1:L
    for (unsigned int k=0; k < L; ++k)
    {
        fmat XColK = X.col(k);
        Z1.col(k)= process->ffun(&XColK);
        z1=z1+Wm(k)*Z1.col(k);
    }

    //Z2=Z1-x1(:,ones(1,L));
    Z2 = Z1;
    for (unsigned int j = 0; j < L; ++j)
    {
        for (unsigned int i = 0; i < x1.n_rows; ++i)
        {
            Z2(i,j) -= x1(i);
        }
    }

    P2=Z2*Wc.diag()*Z2.t()+R;
}



fmat BFilterUKF::sigmas(fvec x, fmat P, float c)
{
    //Sigma points around reference point
    //Inputs:
    //       x: reference point
    //       P: covariance
    //       c: coefficient
    //Output:
    //       X: Sigma points
    fmat X;

    fmat cholP = chol(P);
    fmat A = c*cholP.t();

    // Y = x(:,ones(1,numel(x)));
    fmat Y = zeros<fmat>(x.n_rows, x.n_elem);
    for (unsigned int j = 0; j < x.n_elem; ++j)
    {
        for (unsigned int i = 0; i < x.n_rows; ++i)
        {
            Y(i,j) = x1(i);
        }
    }

    //X = [x Y+A Y-A];
    X = fmat(x);
    X.insert_cols(X.n_cols,Y+A);
    X.insert_cols(X.n_cols,Y-A);

    return X;
}


BFilterUKF::~BFilterUKF(){
}
