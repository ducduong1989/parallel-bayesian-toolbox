#include "mex.h"
#define USE_OCTAVE
#ifdef USE_OCTAVE
//#include "config.h"
#endif
//#include <Matrix.h>
#include <string>

#include "bf_sirpf.h"
#include "estimation_mean.h"
#include "estimation_median.h"
#include "resampling_multinomial.h"
#include "resampling_systematic.h"
#include "resampling_stratified.h"
#include "resampling_residual.h"

#include "matlab_interface.h"

#include "model_linear/model_linear.h"
#include "model_straight3d/model_straight3d.h"
#include "model_fancy/model_fancy.h"
#include "model_sinus/model_sinus.h"
#include "model_car_localization/model_car_localization.h"
#include "model_wpam_3d_trajectory/model_wpam_3d.h"

static PFilter *filter;
static Resampling *resampler;
static Estimation *estimator;
static Model *process;
int number;
int dimension;

using namespace std;

void printHelpText(){
    mexPrintf("Function call: assume library name is pbtmex (in MATLAB) or pbtoct (in OCTAVE)\n");
    mexPrintf("\tresult = pbtmex('key' [, matrix] [, matrix])\n");
    mexPrintf("Possible call combinations:\n");
    mexPrintf(" pbtmex('initialize')              - initializes all needed things\n");
	mexPrintf(" pbtmex('setModel','usedModel')    - changes the uses model\n");
	mexPrintf(" pbtmex('getModels')               - changes the uses model\n");
	mexPrintf(" pbtmex('getModelDescription')     - changes the uses model\n");
    mexPrintf(" pbtmex('getCovariance')           - writes the covariance to the struct\n");
    mexPrintf(" pbtmex('getParticles')            - writes the particles to the struct\n");
    mexPrintf(" pbtmex('predict')                 - does prediction step\n");
    mexPrintf(" pbtmex('update', z)               - does measurement update step using z\n");
    mexPrintf(" pbtmex('setParticles', s, w)      - replace sample set with s and weights with w\n");
    mexPrintf(" pbtmex('setThresholdByNumber', t) - sets resampling threshold as a number\n");
    mexPrintf(" pbtmex('setThresholdByFactor', t) - sets resampling threshold as a factor\n");
    mexPrintf(" pbtmex('cleanup')                 - clears the memory\n");
    mexPrintf(" pbtmex('help')                    - shows this text\n");
    mexPrintf("Returns a struct with fields:\n");
    mexPrintf(" error      - is not zero if an error occurs\n");
    mexPrintf(" estimation - current estimation, updates after every prediction and update step\n");
    mexPrintf(" samples    - sample set; each column represents a  particle state vector\n");
    mexPrintf(" weights    - weights of samples/ particles\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    bool computed = false;

    if (nrhs < 1)
    {
        mexErrMsgTxt("Incorrect number of input arguments\n");
        printHelpText();
    }
    if (nlhs != 1)
    {
        mexErrMsgTxt("Incorrect number of output arguments\n");
        printHelpText();
    }

    // initialize helper function object
    MatlabInterface pbtInterface;

    // initialize mexArrays of output struct
    mwSize dims[2] = {1, 1};
    mxArray *estimation = mxCreateDoubleMatrix(dimension,1,mxREAL);
    mxArray *covariance = mxCreateDoubleMatrix(dimension,dimension,mxREAL);
    mxArray *error = mxCreateDoubleMatrix(1,1,mxREAL);
    mxArray *samples = mxCreateDoubleMatrix(1,1,mxREAL);
    mxArray *weights = mxCreateDoubleMatrix(1,1,mxREAL);

    //initialize the armadillo matricies and link it to mexArrays
    mat errorPBT = zeros<mat>(1,1);
    const double* errorMem=access::rw(errorPBT.mem);
    pbtInterface.matlab2arma(errorPBT,error);

    mat estimationPBT = mat(dimension,1);
    const double* estimationMem=access::rw(estimationPBT.mem);
    pbtInterface.matlab2arma(estimationPBT,estimation);

    mat covariancePBT = mat(dimension,dimension);
    const double* covarianceMem=access::rw(covariancePBT.mem);
    pbtInterface.matlab2arma(covariancePBT,covariance);

    mat samplesPBT = mat(dimension,number);
    const double* samplesMem=access::rw(samplesPBT.mem);

    mat weightsPBT = mat(1,number);
    const double* weightsMem=access::rw(weightsPBT.mem);

    // construct the struct field
    const char *keys[] = { "error", "estimation", "covariance", "particles", "weights" };
    plhs[0] = mxCreateStructArray (2, dims, 5, keys);

    // get command string from matlab and save it in a string
    char* command = new char[100];
    mxGetString(prhs[0],command,100);
    mexPrintf("\nGot: %s\n", command);
    string commandString(command);

    if (commandString.find("help")!= string::npos)
    {
        computed = true;
        printHelpText();
    }

    if (commandString.find("initialize")!= string::npos)
    {
        computed = true;
        filter = new BFilterSIR;
        resampler = new ResamplingMultinomial;
        estimator = new EstimationMean;
        process = new ModelWPAM;

        // specify estimator and resampler that are used for particle filter
        filter->setResampling(resampler);
        filter->setEstimation(estimator);
        filter->setModel(process);
        dimension = process->getProcessDimension();

        mexPrintf("PBT initialized\n");
    }

	if (commandString.find("setModel")!= string::npos)
    {
		computed = true;
        char* method = new char[100];
        mxGetString(prhs[1],command,100);
        string methodString(method);

		delete process;

		if( methodString.find("linear")!= string::npos)    
		{
			ModelLinear* temp = new ModelLinear;
			process = (Model*) temp;
		}
        if( methodString.find("straight")!= string::npos)
        {
            ModelStraight3D* temp = new ModelStraight3D;
            fvec origin = zeros<fvec>(3);
            fvec vector = zeros<fvec>(3);
            vector(0) = 1;
            vector(1) = 1;
            vector(2) = 1;
            temp->addStraightDefinition(origin, vector);
            process = (Model*) temp;
        }
        if(methodString.find("sinus")!= string::npos)       
		{
			ModelSinus* temp = new ModelSinus;
			process = (Model*) temp;
		}
        if( methodString.find("fancy")!= string::npos)      
		{
			ModelFancy* temp = new ModelFancy;
			process = (Model*) temp;
		}
		if( methodString.find("wpam")!= string::npos)     
		{
			ModelWPAM* temp = new ModelWPAM;
			process = (Model*) temp;
		}
		if( methodString.find("car")!= string::npos)      
		{
			ModelCarLocalization* temp = new ModelCarLocalization;
			process = (Model*) temp;
		}

		filter->setModel( process);

        mexPrintf("PBT model is set\n");
    }

	if (commandString.find("getModels")!= string::npos)
    {
        computed = true;
		mexPrintf("Included models are:\n\n");
		// List of implemented models
		mexPrintf("linear\n");
        mexPrintf("straight");
		mexPrintf("sinus\n");
		mexPrintf("fancy\n");
		mexPrintf("wpam\n");
		mexPrintf("car\n");

        mexPrintf("\n");
    }

	if (commandString.find("getModelDescription")!= string::npos)
    {
	   computed = true;
       mexPrintf("Current model description %s\n", process->descriptionTag.c_str());
    }

    if (commandString.find("setResamplingMethod")!= string::npos)
    {
        computed = true;
        char* method = new char[100];
        mxGetString(prhs[1],command,100);
        string methodString(method);

        if( methodString.find("multinomial")!= string::npos)     resampler = new ResamplingMultinomial;
        if(methodString.find("systematic")!= string::npos)       resampler = new ResamplingSystematic;
        if( methodString.find("stratified")!= string::npos)      resampler = new ResamplingStratified;
		if( methodString.find("residual")!= string::npos)        resampler = new ResamplingResidual;

        filter->setResampling(resampler);

        mexPrintf("PBT resampling method is set\n");
    }

    if (commandString.find("setEstimationMethod")!= string::npos)
    {
        computed = true;
        char* method = new char[100];
        mxGetString(prhs[1],command,100);
        string methodString(method);

        if( methodString.find("mean")!= string::npos)            estimator = new EstimationMean;
        if( methodString.find("median")!= string::npos)          estimator = new EstimationMedian;

        filter->setEstimation(estimator);

        mexPrintf("PBT  estimation method is set\n");
    }

    if (commandString.find("setParticles")!= string::npos)
    {
        computed = true;

        mat inputSamples(1,1);
        const double* inputSamplesMem=access::rw(inputSamples.mem);
        pbtInterface.matlab2arma(inputSamples,prhs[1]); // First create the matrix, then change it to point to the matlab data.

        mat inputWeights(1,1);
        const double* inputWeightsMem=access::rw(inputWeights.mem);
        pbtInterface.matlab2arma(inputWeights,prhs[2]); // First create the matrix, then change it to point to the matlab data.

        fmat temp(pbtInterface.DoubleToFloatArmaMat(inputSamples));

        fmat samples(temp);

        fmat temp2(pbtInterface.DoubleToFloatArmaMat(inputWeights));

        frowvec weights(temp2);

        filter->setSamples(samples);
        filter->setWeights(weights);

        estimationPBT = (pbtInterface.floatToDoubleArmaMat(filter->getEstimation()));
        estimationMem=access::rw(estimationPBT.mem);
        pbtInterface.matlab2arma(estimationPBT,estimation);


        pbtInterface.freeVar(inputSamples,inputSamplesMem); // Change back the pointers!!
        pbtInterface.freeVar(inputWeights,inputWeightsMem); // Change back the pointers!!

        mexPrintf("PBT particles set\n");
    }

    if (commandString.find("setThresholdByNumber")!= string::npos)
    {
        computed = true;

        mat inputThreshold(1,1);
        const double* inputThresholdMem=access::rw(inputThreshold.mem);
        pbtInterface.matlab2arma(inputThreshold,prhs[1]); // First create the matrix, then change it to point to the matlab data.

        fmat temp = (pbtInterface.DoubleToFloatArmaMat(inputThreshold));

        filter->setThresholdByNumber((int)floor(temp(0,0)));

        pbtInterface.freeVar(inputThreshold,inputThresholdMem); // Change back the pointers!!

        mexPrintf("PBT threshold set\n");
    }

    if (commandString.find("setThresholdByFactor")!= string::npos)
    {
        computed = true;

        mat inputThreshold(1,1);
        const double* inputThresholdMem=access::rw(inputThreshold.mem);
        pbtInterface.matlab2arma(inputThreshold,prhs[1]); // First create the matrix, then change it to point to the matlab data.

        fmat temp(pbtInterface.DoubleToFloatArmaMat(inputThreshold));

        filter->setThresholdByFactor(temp(0,0));

        pbtInterface.freeVar(inputThreshold,inputThresholdMem); // Change back the pointers!!

        mexPrintf("PBT threshold set\n");
    }

    if (commandString.find("predict")!= string::npos)
    {
        computed = true;
        filter->predict();
        mexPrintf("PBT prediction step performed\n");

        fmat ftemp = fmat(filter->getEstimation());
        mat temp = pbtInterface.floatToDoubleArmaMat(ftemp);
        estimationPBT = mat(temp);
        estimationMem=access::rw(estimationPBT.mem);
        pbtInterface.matlab2arma(estimationPBT,estimation);

        mexPrintf("PBT prediction step performed\n");
    }

    if (commandString.find("update")!= string::npos)
    {
        computed = true;

        mat input(1,1);
        const double* inputMem=access::rw(input.mem);
        pbtInterface.matlab2arma(input,prhs[1]); // First create the matrix, then change it to point to the matlab data.

        fmat temp = pbtInterface.DoubleToFloatArmaMat(input);
        fvec measurement(temp);

        filter->update(measurement);

        fmat ftemp = fmat(filter->getEstimation());
        mat temp3 = pbtInterface.floatToDoubleArmaMat(ftemp);
        estimationPBT = mat(temp3);
        estimationMem=access::rw(estimationPBT.mem);
        pbtInterface.matlab2arma(estimationPBT,estimation);

        pbtInterface.freeVar(input,inputMem); // Change back the pointers!!

        mexPrintf("PBT update performed\n");
    }

    if (commandString.find("getParticles")!= string::npos)
    {
        computed = true;
        Particles particles = filter->getParticles();
        samplesPBT = mat(particles.samples.n_rows,particles.samples.n_cols);
        samplesMem=access::rw(samplesPBT.mem);
        weightsPBT = mat(1,particles.weights.n_cols);
        weightsMem=access::rw(weightsPBT.mem);
        samples = mxCreateDoubleMatrix(samplesPBT.n_rows,samplesPBT.n_cols,mxREAL);
        weights = mxCreateDoubleMatrix(weightsPBT.n_rows,weightsPBT.n_cols,mxREAL);
        pbtInterface.matlab2arma(samplesPBT,samples);
        pbtInterface.matlab2arma(weightsPBT,weights);

        for (unsigned int i=0; i< particles.weights.n_cols; ++i)
        {
            weightsPBT(i) = particles.weights(i);
            for (unsigned int j=0; j<particles.samples.n_rows;++j)
            {
                samplesPBT(j,i) = particles.samples(j,i);
            }
        }


        mexPrintf("PBT Particles set as output\n");
    }

    if (commandString.find("getCovariance")!= string::npos)
    {
        computed = true;

        fmat ftemp = fmat(filter->getCovariance());
        mat temp3 = pbtInterface.floatToDoubleArmaMat(ftemp);
        covariancePBT = mat(temp3);
        covarianceMem=access::rw(covariancePBT.mem);
        pbtInterface.matlab2arma(covariancePBT,covariance);

        mexPrintf("PBT Covariance set as output\n");
    }

    if (commandString.find("cleanup")!= string::npos)
    {
        computed = true;
        delete filter;
        delete resampler;
        delete estimator;
        delete process;
        mexPrintf("PBT is deleted\n");
        return;
    }

    if (computed == false)
    {
        printHelpText();
    }

    mxSetField (plhs[0], 0, "error" , error);
    mxSetField (plhs[0], 0, "estimation" , estimation);
    mxSetField (plhs[0], 0, "covariance" , covariance);
    mxSetField (plhs[0], 0, "particles" , samples);
    mxSetField (plhs[0], 0, "weights" , weights);


    pbtInterface.freeVar(errorPBT,errorMem);
    pbtInterface.freeVar(estimationPBT,estimationMem);
    pbtInterface.freeVar(covariancePBT,covarianceMem);
    pbtInterface.freeVar(samplesPBT,samplesMem);
    pbtInterface.freeVar(weightsPBT,weightsMem);

}

void mexExitFunction(void){

}
