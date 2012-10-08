#include "noise_gauss_gpu.h"
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265
#endif
#include <cuda.h>
#include <cuda_runtime.h>

void callGaussEvaluationV(float* x, int number, float mean, float stddev, float* result);

void callGaussEvaluationM(float* x, int number, int dim, int numberOfDims, float mean, float stddev, float* result);


frowvec NoiseGaussianGPU::sample(unsigned int number){
    frowvec samples;

    samples = a + randn<frowvec>(number) *b ;

    return samples;
}

frowvec NoiseGaussianGPU::eval(frowvec input){
    int num = input.n_cols;
    frowvec result = zeros<frowvec>(num);

    float* devX;
    float* devResult;
    cudaError_t cudaStat;

    cudaStat = cudaMalloc((void**)&devX, num * sizeof(float));
    if ( cudaStat != cudaSuccess )
    {
        perror ( "device memory allocation failed\n" ) ;
    }

    cudaStat = cudaMalloc((void**)&devResult, num * sizeof(float));
    if ( cudaStat != cudaSuccess )
    {
        perror ( "device memory allocation failed\n" ) ;
    }

    cudaMemcpy(devX,input.memptr(),(size_t) input.n_elem * sizeof(float), cudaMemcpyHostToDevice);

    callGaussEvaluationV(devX, num, a, b, devResult);

    cudaMemcpy(result.memptr(),devResult,result.n_elem * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devX);
    cudaFree(devResult);

    return result;
}

frowvec NoiseGaussianGPU::eval(float* input, int number, int dim, int numberOfDims){
    frowvec result = zeros<frowvec>(number);
	
    float* devResult;
    cudaError_t cudaStat;

    cudaStat = cudaMalloc((void**)&devResult, number * sizeof(float));
    if ( cudaStat != cudaSuccess )
    {
        perror ( "device memory allocation failed\n" ) ;
	}
	//else printf ( "device memory allocation passed\n" ) ;

	callGaussEvaluationM(devResult, number, dim, numberOfDims, a, b, devResult);

    cudaMemcpy(result.memptr(),devResult,result.n_elem * sizeof(float), cudaMemcpyDeviceToHost);
	if ( cudaStat != cudaSuccess )
    {
        perror ( "copying memory to host failed\n" ) ;
	}
	//else printf ( "copying memory to host passed\n\n" ) ;

    cudaFree(devResult);

    return result;
}

NoiseGaussianGPU::NoiseGaussianGPU(float mean, float stddev){
    a = mean;
    b= stddev;
}

NoiseGaussianGPU::NoiseGaussianGPU(const NoiseGaussianGPU &other){
    a = other.a;
    b= other.b;
}


NoiseGaussianGPU::~NoiseGaussianGPU(){

}
