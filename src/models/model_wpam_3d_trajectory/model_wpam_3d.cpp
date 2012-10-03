#include "model_wpam_3d.h"
#include "noise_gauss.h"
#include <math.h>

void ModelWPAM::setDescriptionTag(){
    descriptionTag = "WPAM/DWPA";
	// DWPA (see Bar-Shalom, p.274):  x(k+1) = F*x(k) + G*v(k) <-- pNoise Q
	//							      y(k) = C*x(k) + H*w(k) <-- oNoise; C = I
}

   //void ModelWPAM::setPNoise(){
   //    NoiseGaussian* sx = new NoiseGaussian(0.0f, 50e-6f);
   //    pNoise.addNoise(sx);
   //    NoiseGaussian* sy = new NoiseGaussian(0.0f, 50e-6f);
   //    pNoise.addNoise(sy);
   //    NoiseGaussian* sz = new NoiseGaussian(0.0f, 50e-6f);
   //    pNoise.addNoise(sz);

   //    NoiseGaussian* vx = new NoiseGaussian(0.0f, 10e-6f);
   //    pNoise.addNoise(vx);
   //    NoiseGaussian* vy = new NoiseGaussian(0.0f, 10e-6f);
   //    pNoise.addNoise(vy);
   //    NoiseGaussian* vz = new NoiseGaussian(0, 10e-6f);
   //    pNoise.addNoise(vz);

   //    NoiseGaussian* ax = new NoiseGaussian(0.0f, 100e-6f);
   //    pNoise.addNoise(ax);
   //    NoiseGaussian* ay = new NoiseGaussian(0.0f, 100e-6f);
   //    pNoise.addNoise(ay);
   //    NoiseGaussian* az = new NoiseGaussian(0.0f, 100e-6f);
   //    pNoise.addNoise(az);


   // }

    void ModelWPAM::setONoise(){
        fvec mean = zeros<fvec>(dimmeas*3);
        fvec var = ones<fvec>(dimmeas*3);
        var = var*(5e-2f*5e-2f);

        W = eye<fmat>(9,9);

        for (unsigned int i=0; i< mean.n_rows;i++)
        {
            NoiseGaussian* temp = new NoiseGaussian( mean(i), var(i));
            W(i,i)=var(i);
            oNoise.addNoise(temp);
        }
    }

    //void ModelWPAM::setUNoise(){
	void ModelWPAM::setPNoise(){
        fmat E = eye<fmat>(3,3);
        fmat Q_Row(0,0);
        Q = zeros<fmat>(0,0);

        Q_Row.insert_cols(0,(pow(T,4)/4)*E);
        Q_Row.insert_cols(3,(pow(T,3)/2)*E);
        Q_Row.insert_cols(6,(pow(T,2)/2)*E);

        Q.insert_rows(0,Q_Row);

        Q_Row = fmat(0,0);
        Q_Row.insert_cols(0,(pow(T,3)/2)*E);
        Q_Row.insert_cols(3,(pow(T,2))*E);
        Q_Row.insert_cols(6,T*E);

        Q.insert_rows(3,Q_Row);

        Q_Row = fmat(0,0);
        Q_Row.insert_cols(0,(pow(T,2)/2)*E);
        Q_Row.insert_cols(3,T*E);
        Q_Row.insert_cols(6,E);

        Q.insert_rows(6,Q_Row);

        for (unsigned int i=0; i< Q.n_cols;++i)
        {
            NoiseGaussian* u = new NoiseGaussian(0,Q(i,i)*variance);
            pNoise.addNoise(u);
        }
        Q = Q*variance;
    }

    fmat ModelWPAM::ffun(fmat *current){

        fmat prediction = zeros<fmat>(current->n_rows,current->n_cols);
        fmat pNoiseSample = pNoise.sample(current->n_cols);
        //fmat u = U.sample(current->n_cols);
        //prediction = F * (*current) + pNoiseSample + u ;
        prediction = (F* fmat(*current));
        prediction = prediction + pNoiseSample;
        return prediction;
    }

    fmat ModelWPAM::hfun(fmat *state){

        fmat measurement(state->n_rows,state->n_cols);
        fmat oNoiseSample = oNoise.sample(state->n_cols);

        measurement = *state + oNoiseSample;

        return measurement;
    }


    void ModelWPAM::setF(){
        F = eye<fmat>(9,9);

        //fmat A = eye<fmat>(dimmeas*3,dimmeas*3);

        //fmat B = diagmat(T * ones<fmat>(1),dimmeas*2);

        //fmat C = diagmat((pow(T,2)/2)*ones<fmat>(1),dimmeas);

        //F = A + B + C;

        F(0,3) = T;
        F(1,4) = T;
        F(2,5) = T;
        F(3,6) = T;
        F(4,7) = T;
        F(5,8) = T;

        F(0,6) = (T*T)/2;
        F(1,7) = (T*T)/2;
        F(2,8) = (T*T)/2;

    }

    void ModelWPAM::setH(){
        H = eye<fmat>(9,9);
    }

    ModelWPAM::ModelWPAM(){
        variance = (100e-3f*100e-3f);
        dimmeas = 3;
        T = 1.0f/30.0f;

        setF();
    }

    ModelWPAM::~ModelWPAM(){

    }
