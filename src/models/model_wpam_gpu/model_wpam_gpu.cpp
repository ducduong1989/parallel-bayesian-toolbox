#include "model_wpam_gpu.h"
#include "noise_gauss_gpu.h"
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

void ModelWPAMGPU::setDescriptionTag(){
    descriptionTag = "WPAM on GPU";
}

frowvec ModelWPAMGPU::eval_gpu(float* values, int numberOfParticles)
{
	frowvec evaluation;
	//printf("start evaluation ...\n");
    evaluation = pNoise.eval(values, numberOfParticles);

    return evaluation;
}

void ModelWPAMGPU::setPNoise()
{
    NoiseGaussianGPU* sx = new NoiseGaussianGPU(0.0f, 50e-6f);
    pNoise.addNoise(sx);
    NoiseGaussianGPU* sy = new NoiseGaussianGPU(0.0f, 50e-6f);
    pNoise.addNoise(sy);
    NoiseGaussianGPU* sz = new NoiseGaussianGPU(0.0f, 50e-6f);
    pNoise.addNoise(sz);

    NoiseGaussianGPU* vx = new NoiseGaussianGPU(0.0f, 10e-6f);
    pNoise.addNoise(vx);
    NoiseGaussianGPU* vy = new NoiseGaussianGPU(0.0f, 10e-6f);
    pNoise.addNoise(vy);
    NoiseGaussianGPU* vz = new NoiseGaussianGPU(0, 10e-6f);
    pNoise.addNoise(vz);

    NoiseGaussianGPU* ax = new NoiseGaussianGPU(0.0f, 100e-6f);
    pNoise.addNoise(ax);
    NoiseGaussianGPU* ay = new NoiseGaussianGPU(0.0f, 100e-6f);
    pNoise.addNoise(ay);
    NoiseGaussianGPU* az = new NoiseGaussianGPU(0.0f, 100e-6f);
    pNoise.addNoise(az);
}

void ModelWPAMGPU::setONoise()
{
    fvec mean = zeros<fvec>(dimmeas*3);
    fvec var = ones<fvec>(dimmeas*3) * 50.0e-3f;

    for (unsigned int i=0; i< mean.n_rows;i++)
    {
        NoiseGaussianGPU* temp = new NoiseGaussianGPU( mean(i), var(i));
        oNoise.addNoise(temp);
    }
}

void ModelWPAMGPU::setUNoise()
{
    fmat E = eye<fmat>(3,3);
    fmat Q_Row(0,0);
    fmat Q(9,9);

    Q_Row.insert_cols(0,(pow(T,4)/4)*E);
    Q_Row.insert_cols(3,(pow(T,3)/2)*E);
    Q_Row.insert_cols(6,(pow(T,2)/2)*E);

    Q.insert_rows(0,Q_Row);

    Q_Row = fmat(0,0);
    Q_Row.insert_cols(0,(pow(T,3)/2)*E);
    Q_Row.insert_cols(3,(pow(T,2))*E);
    Q_Row.insert_cols(6,T*E);

    Q.insert_rows(0,Q_Row);

    Q_Row = fmat(0,0);
    Q_Row.insert_cols(0,(pow(T,2)/2)*E);
    Q_Row.insert_cols(3,T*E);
    Q_Row.insert_cols(6,E);

    Q.insert_rows(0,Q_Row);

    for (unsigned int i=0; i< Q.n_cols;++i)
    {
        NoiseGaussianGPU* u = new NoiseGaussianGPU(0,Q(i,i)*variance);
        U.addNoise(u);
    }
}

fmat ModelWPAMGPU::ffun(fmat *current)
{
    fmat prediction(current->n_rows,current->n_cols);
    fmat pNoiseSample = pNoise.sample(current->n_cols);
    fmat u = U.sample(current->n_cols);
    float* lastState_dev;
    float* F_dev;
	float* U_dev;
	float* pNoise_dev;
    int stateDimension = current->n_rows;
    int numberOfSamples = current->n_cols;
    float* newState_dev;

	//allocate memory on gpu
    cudaMalloc( &lastState_dev, (size_t) current->n_elem * sizeof(float)) ;
	cudaMalloc( &F_dev, (size_t) F.n_elem * sizeof(float)) ;
	cudaMalloc( &U_dev, (size_t) u.n_elem * sizeof(float)) ;
	cudaMalloc( &pNoise_dev, (size_t) pNoiseSample.n_elem * sizeof(float)) ;
	cudaMalloc( &newState_dev, (size_t) prediction.n_elem * sizeof(float)) ;

	//Copy particles and weights to the gpu
    cudaMemcpy(lastState_dev,current->memptr(),(size_t) current->n_elem * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(F_dev,F.memptr(),(size_t) F.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(U_dev,u.memptr(),(size_t) u.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(pNoise_dev,pNoiseSample.memptr(),(size_t) pNoiseSample.n_elem * sizeof(float), cudaMemcpyHostToDevice);

    //pNoise
    curandGenerateNormal(gen, pNoise_dev, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+2*numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+3*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+4*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+5*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+6*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+7*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+8*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);

    // U
    U.batch.at(0);
    for (unsigned int i=0; i< 9 ;++i)
    {
        curandGenerateNormal(gen, U_dev+ i*numberOfSamples, numberOfSamples, U.batch.at(i)->a, U.batch.at(i)->b);
    }
    /*curandGenerateNormal(gen, oNoise_dev, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+2*numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+3*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+4*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+5*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+6*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+7*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+8*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);*/

    //prediction = F * current + pNoiseSample + u ;
    callFfunKernel(lastState_dev, F_dev, U_dev, pNoise_dev, stateDimension ,numberOfSamples,newState_dev);
    //printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	//get estimation from gpu
    cudaMemcpy(prediction.memptr(),newState_dev,current->n_elem * sizeof(float), cudaMemcpyDeviceToHost);

	// clean up the graphics card
    cudaFree(lastState_dev);
	cudaFree(newState_dev);
	cudaFree(F_dev);
	cudaFree(U_dev);
	cudaFree(pNoise_dev);

    return prediction;
}

float* ModelWPAMGPU::ffun_gpu(fmat *current)
{
    //fmat pNoiseSample = pNoise.sample(current.n_cols);
    //fmat u = U.sample(current.n_cols);
    float* lastState_dev;
    float* F_dev;
	float* U_dev;
	float* pNoise_dev;
    int stateDimension = current->n_rows;
    int numberOfSamples = current->n_cols;
    float* newState_dev;

	//allocate memory on gpu
    cudaMalloc( &lastState_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;
    cudaMalloc( &F_dev, (size_t) F.n_elem * sizeof(float)) ;
    cudaMalloc( &U_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;
    cudaMalloc( &pNoise_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;
    cudaMalloc( &newState_dev, (size_t) stateDimension*numberOfSamples * sizeof(float)) ;

	//Copy particles and weights to the gpu
    cudaMemcpy(lastState_dev,current->memptr(),(size_t) current->n_elem * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(F_dev,F.memptr(),(size_t) F.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(U_dev,u.memptr(),(size_t) u.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(pNoise_dev,pNoiseSample.memptr(),(size_t) pNoiseSample.n_elem * sizeof(float), cudaMemcpyHostToDevice);

	//pNoise
    curandGenerateNormal(gen, pNoise_dev, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+2*numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+3*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+4*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+5*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+6*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+7*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, pNoise_dev+8*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);

	// U
    U.batch.at(0);
    for (unsigned int i=0; i< 9 ;++i)
    {
        curandGenerateNormal(gen, U_dev+ i*numberOfSamples, numberOfSamples, U.batch.at(i)->a, U.batch.at(i)->b);
    }
    /*curandGenerateNormal(gen, oNoise_dev, numberOfSamples, 0.0f, 50.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+2*numberOfSamples, numberOfSamples, 0.0f, 50.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+3*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+4*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+5*numberOfSamples, numberOfSamples, 0.0f, 10.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+6*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
	curandGenerateNormal(gen, oNoise_dev+7*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);
    curandGenerateNormal(gen, oNoise_dev+8*numberOfSamples, numberOfSamples, 0.0f, 100.0e-6f);*/

    //prediction = F * current + pNoiseSample + u ;
    callFfunKernel(lastState_dev, F_dev, U_dev, pNoise_dev, stateDimension ,numberOfSamples,newState_dev);
	//printf("ffun error: %s\n",cudaGetErrorString(cudaGetLastError()));

	//get estimation from gpu
    //cudaMemcpy(prediction.memptr(),newState_dev,current.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

	// clean up the graphics card
    cudaFree(lastState_dev);
	//cudaFree(newState_dev);
	cudaFree(F_dev);
	cudaFree(U_dev);
	cudaFree(pNoise_dev);

    return newState_dev;
}

fmat ModelWPAMGPU::hfun(fmat *state)
{
	float* state_dev;
    float* oNoise_dev;
    int stateDimension = state->n_rows;
    int numberOfSamples = state->n_cols;
    float* meas_dev;

    fmat measurement(state->n_rows,state->n_cols);
    //fmat oNoiseSample = oNoise.sample(state->n_cols);

    //measurement = state + oNoiseSample;

	//allocate memory on gpu
    cudaMalloc( &state_dev, (size_t) state->n_elem * sizeof(float)) ;
    cudaMalloc( &oNoise_dev, (size_t) numberOfSamples * stateDimension * sizeof(float)) ;
    cudaMalloc( &meas_dev, (size_t) numberOfSamples * stateDimension * sizeof(float)) ;

	//Copy particles and weights to the gpu
    cudaMemcpy(state_dev,state->memptr(),(size_t) state->n_elem * sizeof(float), cudaMemcpyHostToDevice);

    //generate random particles
    //cudaMemcpy(oNoise_dev,oNoiseSample.memptr(),(size_t) oNoiseSample.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    curandGenerateNormal(gen, oNoise_dev, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+2*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+3*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+4*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+5*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+6*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+7*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
    curandGenerateNormal(gen, oNoise_dev+8*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);


    //prediction = F * current + pNoiseSample + u ;
	callHfunKernel(state_dev,oNoise_dev,stateDimension,numberOfSamples,meas_dev);
    //printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	//get estimation from gpu
	cudaMemcpy(measurement.memptr(),meas_dev,measurement.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

	// clean up the graphics card
    cudaFree(state_dev);
	cudaFree(oNoise_dev);
	cudaFree(meas_dev);

    return measurement;
}

float* ModelWPAMGPU::hfun_gpu(float* values, int numberOfSamples, int stateDimension)
{
	
    float* oNoise_dev;
    float* meas_dev;
	
	

    //fmat oNoiseSample = oNoise.sample(numberOfSamples);

    //measurement = state + oNoiseSample;

	//allocate memory on gpu
	cudaMalloc( &oNoise_dev, (size_t) numberOfSamples * stateDimension * sizeof(float)) ;
	cudaMalloc( &meas_dev, (size_t) numberOfSamples * stateDimension * sizeof(float)) ;

	//generate random particles
	//cudaMemcpy(oNoise_dev,oNoiseSample.memptr(),(size_t) oNoiseSample.n_elem * sizeof(float), cudaMemcpyHostToDevice);
	curandGenerateNormal(gen, oNoise_dev, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+2*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+3*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+4*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+5*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+6*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+7*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);
	curandGenerateNormal(gen, oNoise_dev+8*numberOfSamples, numberOfSamples, 0.0f, 50.0e-3f);

    //prediction = F * current + pNoiseSample + u ;
	callHfunKernel(values,oNoise_dev,stateDimension,numberOfSamples,meas_dev);
    //printf("%s\n",cudaGetErrorString(cudaGetLastError()));

	//get estimation from gpu
	//cudaMemcpy(measurement.memptr(),meas_dev,measurement.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

	// clean up the graphics card
	cudaFree(oNoise_dev);
	cudaFree(values);
	
	return meas_dev;
}


void ModelWPAMGPU::addF()
{
    F = eye<fmat>(9,9);

    //fmat A = eye<fmat>(dimmeas*3,dimmeas*3);

    //fmat B = diagmat(T * ones<fmat>(1),dimmeas*2);

    //fmat C = diagmat((pow(T,2)/2)*ones<fmat>(1),dimmeas);

    //F = A + B + C;

    F(3,0) = T;
    F(4,1) = T;
    F(5,2) = T;
    F(6,3) = T;
    F(7,4) = T;
    F(8,5) = T;

    F(5,0) = (T*T)/2;
    F(6,1) = (T*T)/2;
    F(7,2) = (T*T)/2;

}

ModelWPAMGPU::ModelWPAMGPU()
{
    variance = 100e-6f;
    dimmeas = 3;
    T = 1.0f/30.0f;

    addF();

    setUNoise();

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

ModelWPAMGPU::~ModelWPAMGPU()
{
	curandDestroyGenerator(gen);
}
