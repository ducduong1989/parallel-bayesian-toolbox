#include <armadillo>
#include <hr_time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "machine_info.h"
#include "model_wpam_3d_trajectory/model_wpam_3d.h"
#include "model_wpam_gpu/model_wpam_gpu.h"

using namespace arma;

void callCartToPolarTransformation(float* x_dev, int dim, int num, float* result_dev);
void callPolarToCartTransformation(float* x_dev, int dim, int num, float* result_dev);
int runMultTest(int dim, int num);
int runTransformationTest(int num);
int runWPAMTest(int num);

void printCUBLASError(cublasStatus_t error){
	if (error == CUBLAS_STATUS_NOT_INITIALIZED) printf("CUBLAS_STATUS_NOT_INITIALIZED\n");
	if (error == CUBLAS_STATUS_ALLOC_FAILED) printf("CUBLAS_STATUS_ALLOC_FAILED\n");
	if (error == CUBLAS_STATUS_INVALID_VALUE) printf("CUBLAS_STATUS_INVALID_VALUE\n");
	if (error == CUBLAS_STATUS_ARCH_MISMATCH) printf("CUBLAS_STATUS_ARCH_MISMATCH\n");
	if (error == CUBLAS_STATUS_MAPPING_ERROR) printf("CUBLAS_STATUS_MAPPING_ERROR\n");
	if (error == CUBLAS_STATUS_EXECUTION_FAILED) printf("CUBLAS_STATUS_EXECUTION_FAILED\n");
	if (error == CUBLAS_STATUS_INTERNAL_ERROR) printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
}

int main (int argc, char* argv[])
{
	int dim=1;
	int num=1;
	int error = 0;

	MachineInformation informations;
	informations.printMachineInformation();

	if (argc != 3)
	{
        printf("MULTIPLICATION TESTS\n");
        runMultTest(3,100);
        runMultTest(3,500);
        runMultTest(3,2000);
        runMultTest(3,10000);

        runMultTest(9,100);
        runMultTest(9,500);
        runMultTest(9,2000);
        runMultTest(9,10000);

        runMultTest(15,100);
        runMultTest(15,500);
        runMultTest(15,2000);
        runMultTest(15,10000);

        printf("TRANSFORMATION TESTS\n");
        runTransformationTest(100);
        runTransformationTest(500);
        runTransformationTest(2000);
        runTransformationTest(10000);

        printf("WPAM TESTS\n");
		runWPAMTest(100);
		runWPAMTest(500);
		runWPAMTest(2000);
		runWPAMTest(10000);
	}
	else {
		dim=atoi(argv[1]);
		num=atoi(argv[2]);
        runMultTest(dim, num);
        runTransformationTest(num);
        runWPAMTest(num);
	}


    // run sorting test

    // run model with coordinate transformation

	return error;
}


int runMultTest(int dim, int num){
    CStopWatch timer;
    int devicecount = 0;
    int device = 0;
    printf("\n\nNumber of dimensions: %i\n",dim);
    printf("Number of particles: %i\n",num);
    unsigned int runs = 400;
    fmat x = randu<fmat>(dim,num);
    fmat y = randu<fmat>(num,dim);
    fmat result = zeros<fmat>(dim,num);

    // CPU test
    timer.startTimer();

    for (unsigned int i=0;i<runs;++i)
    {
        result = x * y;
    }

    timer.stopTimer();

    printf("CPU matrix-matrix multiplication done in %e seconds\n", (double)timer.getElapsedTime()/runs);
    //result.print();

    cudaGetDeviceCount(&devicecount);
    for (device = 0; device < devicecount; ++device)
    {
        cudaSetDevice(device);
        printf("GPU device id %i (no. %i of %i)\n", device, device+1, devicecount );

        // GPU test only one initialization
        //reset result
        result = zeros<fmat>(x.n_rows,y.n_cols);
        timer.startTimer();

        float* devX;
        float* devY;
        float* devResult;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t  handle;
        cublasStatus_t stat;
        cudaError_t cudaStat;

        cudaStat = cudaMalloc((void**)&devX, x.n_rows* x.n_cols * sizeof(float));
        if ( cudaStat != cudaSuccess )
        {
            printf ( "device memory allocation failed\n" ) ;
            return -1;
        }
        cudaStat = cudaMalloc((void**)&devY, y.n_elem * sizeof(float));
        if ( cudaStat != cudaSuccess )
        {
            printf ( "device memory allocation failed\n" ) ;
            return -1;
        }
        cudaStat = cudaMalloc((void**)&devResult, x.n_rows*y.n_cols * sizeof(float));
        if ( cudaStat != cudaSuccess )
        {
            printf ( "device memory allocation failed\n" ) ;
            return -1;
        }
        stat = cublasCreate(&handle);
        if ( stat != CUBLAS_STATUS_SUCCESS) {
            printf ( "CUBLAS initialization failed\n" ) ;
            printCUBLASError(stat);
            return -1 ;
        }


        for (unsigned int i=0;i<runs;++i)
        {
            stat = cublasSetMatrix(x.n_rows, x.n_cols, sizeof(float), x.memptr(), x.n_rows, devX, x.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS host to device transfer failed X\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }
            stat = cublasSetMatrix(y.n_rows, y.n_cols, sizeof(float), y.memptr(), y.n_rows, devY, y.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS host to device transfer failed Y\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }

            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, x.n_rows, y.n_cols, x.n_cols, &alpha, devX, x.n_rows, devY, y.n_rows, &beta, devResult, x.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS calculation failed\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }

            stat = cublasGetMatrix(x.n_rows, y.n_cols, sizeof(float), devResult, x.n_rows, result.memptr(), x.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS device to host transfer failed\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }
        }

        cudaFree(devX);
        cudaFree(devY);
        cudaFree(devResult);
        cublasDestroy(handle);

        timer.stopTimer();

        printf("GPU matrix-matrix multiplication done in %e seconds\n", (double)timer.getElapsedTime()/runs);
        //result.print();


        // GPU test everytime new initialization

        //reset result
        result = zeros<fmat>(x.n_rows,y.n_cols);
        timer.startTimer();



        for (unsigned int i=0;i<runs;++i)
        {
            cudaMalloc((void**)&devX, x.n_elem * sizeof(float));
            cudaMalloc((void**)&devY, y.n_elem * sizeof(float));
            cudaMalloc((void**)&devResult, x.n_rows*y.n_cols * sizeof(float));
            cublasCreate(&handle);

            stat = cublasSetMatrix(x.n_rows, x.n_cols, sizeof(float), x.memptr(), x.n_rows, devX, x.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS host to device transfer failed X\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }
            stat = cublasSetMatrix(y.n_rows, y.n_cols, sizeof(float), y.memptr(), y.n_rows, devY, y.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS host to device transfer failed Y\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }

            stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, x.n_rows, y.n_cols, x.n_cols, &alpha, devX, x.n_rows, devY, y.n_rows, &beta, devResult, x.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS calculation failed Y\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }

            stat = cublasGetMatrix(x.n_rows, y.n_cols, sizeof(float), devResult, x.n_rows, result.memptr(), x.n_rows);
            if ( stat != CUBLAS_STATUS_SUCCESS) {
                printf ( "CUBLAS device to host transfer failed\n" ) ;
                printCUBLASError(stat);
                return -1 ;
            }

            cudaFree(devX);
            cudaFree(devY);
            cudaFree(devResult);
            cublasDestroy(handle);
        }

        timer.stopTimer();

        printf("pessimistic matrix-matrix GPU multiplication done in %e seconds\n", (double)timer.getElapsedTime()/runs);

    }

    device = 0;
    devicecount = 0;
    return 0;
}


int runTransformationTest(int num){
    int dim = 2;
    CStopWatch timer;
    int devicecount = 0;
    int device = 0;
    printf("\n\nNumber of dimensions: %i\n",dim);
    printf("Number of particles: %i\n",num);
    unsigned int runs = 400;
    fmat x = randu<fmat>(dim,num);
    fmat result = zeros<fmat>(dim,num);
    fmat temp;

    // CPU test
    //x.print();
    timer.startTimer();

    for (unsigned int i=0;i<runs;++i)
    {
        for (unsigned j = 0; j < result.n_cols; ++j)
        {
            result(0,j) = sqrt(x(0,j)*x(0,j) + x(1,j)*x(1,j));
            result(1,j) = atan2(x(1,j), x(0,j));

            temp = result;

            result(0,j) = temp(0,j)* cos( temp(1,j) );
            result(1,j) = temp(0,j)* sin( temp(1,j) );

        }
    }

    timer.stopTimer();

    printf("CPU cartesian to polar coord. transformation done in %e seconds\n", (double)timer.getElapsedTime()/runs);
    //result.print();

    cudaGetDeviceCount(&devicecount);
    for (device = 0; device < devicecount; ++device)
    {
        cudaSetDevice(device);
        printf("GPU device id %i (no. %i of %i)\n", device, device+1, devicecount );

        // GPU test only one initialization
        //reset result
        result = zeros<fmat>(dim, num);
        timer.startTimer();

        float* devX;
        float* devResult;
        cudaError_t cudaStat;

        cudaStat = cudaMalloc((void**)&devX, dim* num * sizeof(float));
        if ( cudaStat != cudaSuccess )
        {
            printf ( "device memory allocation failed\n" ) ;
            return -1;
        }

        cudaStat = cudaMalloc((void**)&devResult, dim*num * sizeof(float));
        if ( cudaStat != cudaSuccess )
        {
            printf ( "device memory allocation failed\n" ) ;
            return -1;
        }


        float* temp;
        for (unsigned int i=0;i<runs;++i)
        {
            cudaMemcpy(devX,x.memptr(),(size_t) x.n_elem * sizeof(float), cudaMemcpyHostToDevice);
            callCartToPolarTransformation(devX, dim, num, devResult);
            temp = devX;
            devX = devResult;
            devResult = temp;
            callPolarToCartTransformation(devX, dim, num, devResult);
            cudaMemcpy(result.memptr(),devResult,result.n_elem * sizeof(float), cudaMemcpyDeviceToHost);
        }

        cudaFree(devX);
        cudaFree(devResult);


        timer.stopTimer();

        //result.print();

        printf("GPU cartesian to polar coord. transformation done in %e seconds with %s\n",
               (double)timer.getElapsedTime()/runs, cudaGetErrorString(cudaGetLastError()));
        //result.print();

    }

    device = 0;
    devicecount = 0;
    return 0;
}

int runWPAMTest(int num){
    int dim = 9;
    CStopWatch timer;
    int devicecount = 0;
    int device = 0;
    printf("\n\nNumber of dimensions: %i\n",dim);
    printf("Number of particles: %i\n",num);
    unsigned int runs = 400;
    fmat x = randu<fmat>(dim,num);
    fmat result = zeros<fmat>(dim,num);
	frowvec evaluation = zeros<frowvec>(num);
    fmat temp;

    // CPU test
    //x.print();
    ModelWPAM cpuModel;
    cpuModel.initialize();
    timer.startTimer();
    for (unsigned int i=0;i<runs;++i)
    {
        result = cpuModel.ffun(&x);
        result = cpuModel.hfun(&result);
		evaluation = cpuModel.eval(&result);
    }

    timer.stopTimer();

    printf("CPU WPAM calculation done in %e seconds\n", (double)timer.getElapsedTime()/runs);
    //result.print();

    cudaGetDeviceCount(&devicecount);
    for (device = 0; device < devicecount; ++device)
    {
        cudaSetDevice(device);
        printf("GPU device id %i (no. %i of %i)\n", device, device+1, devicecount );

        // GPU test only one initialization
        //reset result
        result = zeros<fmat>(dim, num);
        ModelWPAMGPU gpgpuModel;
        gpgpuModel.initialize();
        timer.startTimer();

        for (unsigned int i=0;i<runs;++i)
        {
            result = gpgpuModel.ffun(&x);
			
            result = gpgpuModel.hfun(&result);

			evaluation = gpgpuModel.eval(&result);
        }

        timer.stopTimer();

        //result.print();

        printf("GPU WPAM calculation done in %e seconds with %s\n",
               (double)timer.getElapsedTime()/runs, cudaGetErrorString(cudaGetLastError()));
        //result.print();

    }

	cudaGetDeviceCount(&devicecount);
    for (device = 0; device < devicecount; ++device)
    {
        cudaSetDevice(device);
        printf("GPU device id %i (no. %i of %i)\n", device, device+1, devicecount );

        // GPU test only one initialization
        //reset result
        result = zeros<fmat>(dim, num);
        ModelWPAMGPU gpgpuModel;
        gpgpuModel.initialize();
        timer.startTimer();

        for (unsigned int i=0;i<runs;++i)
        {
			float* temp;
            temp = gpgpuModel.ffun_gpu(&x);
			
			temp = gpgpuModel.hfun_gpu(temp,(int)result.n_cols, (int)result.n_rows);
			
			evaluation = gpgpuModel.eval_gpu(temp, (int)result.n_cols);
			
			cudaFree(temp);
        }

        timer.stopTimer();

        //result.print();

        printf("GPU pointer WPAM calculation done in %e seconds with %s\n",
               (double)timer.getElapsedTime()/runs, cudaGetErrorString(cudaGetLastError()));
        //result.print();

    }

    device = 0;
    devicecount = 0;
    return 0;
}
