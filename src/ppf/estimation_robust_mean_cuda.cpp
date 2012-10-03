#include "estimation_robust_mean_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>

fmat EstimationRobustMeanCUDA::getEstimation(Particles* input){
    if (distances.n_rows != input->samples.n_rows){
        perror("EstimationRobustMean configuration error: Dimension of configuration and input doesn't match");
    }
        int mostProbableParticle;
    float max = input->weights(0);
    fvec  estimation = zeros<fvec>(input->samples.n_rows);

    for (unsigned int j=1;j<input->weights.n_cols;++j){
        if (max < input->weights(j)){
            max = input->weights(j);
            mostProbableParticle = j;
        }
    }

    float* dev_weights;
    float* dev_samples;
    float* dev_estimation;
    float* dev_distances;

    //allocate memory on gpu
    cudaMalloc( &dev_samples, (size_t) input->samples.n_elem * sizeof(float)) ;
    cudaMalloc( &dev_weights, (size_t) input->samples.n_cols * sizeof(float)) ;
    cudaMalloc( &dev_distances, (size_t) distances.n_rows * sizeof(float));
    cudaMalloc( &dev_estimation, (size_t) input->samples.n_rows * sizeof(float));

    //Copy particles and weights to the gpu
    cudaMemcpy(dev_weights,input->weights.memptr(),(size_t) input->weights.n_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_samples,input->samples.memptr(),(size_t) input->samples.n_elem * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_samples,distances.memptr(),(size_t) distances.n_rows * sizeof(float), cudaMemcpyHostToDevice);

    //call kernel
    callCalcRobustMeanKernel(dev_samples,dev_weights,dev_distances,input->samples.n_rows,
                             input->samples.n_cols,mostProbableParticle,dev_estimation);

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

void EstimationRobustMeanCUDA::setConfiguration(fvec configuration){
    distances = configuration;
}

EstimationRobustMeanCUDA::~EstimationRobustMeanCUDA(){

}

EstimationRobustMeanCUDA::EstimationRobustMeanCUDA(){
    distances = fvec(1);
}
