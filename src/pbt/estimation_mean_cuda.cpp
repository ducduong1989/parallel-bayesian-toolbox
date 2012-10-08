#include "estimation_mean_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

fmat EstimationMeanCUDA::getEstimation(Particles* input){
    fvec  estimation = zeros<fvec>(input->samples.n_rows);

    float* dev_weights;
    float* dev_samples;
    float* dev_estimation;

    //allocate memory on gpu
    cudaMalloc( &dev_samples, (size_t) input->samples.n_elem * sizeof(float)) ;
    cudaMalloc( &dev_weights, (size_t) input->samples.n_cols * sizeof(float)) ;

    cudaMalloc( &dev_estimation, (size_t) input->samples.n_rows * sizeof(float));

    //Copy particles and weights to the gpu
    cudaMemcpy(dev_weights,input->weights.memptr(),(size_t) input->weights.n_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_samples,input->samples.memptr(),(size_t) input->samples.n_elem * sizeof(float), cudaMemcpyHostToDevice);

    //call kernel
    callCalcMeanKernel(dev_samples,dev_weights,input->samples.n_rows,input->samples.n_cols,dev_estimation);

    //get estimation from gpu
    cudaMemcpy(estimation.memptr(),dev_estimation,input->samples.n_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // clean up the graphics card
    cudaFree(dev_weights);
    cudaFree(dev_samples);
    cudaFree(dev_estimation);

#ifdef VERBOSE
        printf("state estimated \n");
#endif
    return estimation;
}

EstimationMeanCUDA::~EstimationMeanCUDA(){

}

EstimationMeanCUDA::EstimationMeanCUDA(){

}
