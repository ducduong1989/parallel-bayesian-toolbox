#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include "pf_sir.h"
#include "estimation_mean.h"
#include "resampling_multinomial.h"
#include "model_linear.h"

#include "hr_time.h"

int initStateFile(char *file)
{
    std::ofstream out(file,std::ios_base::trunc);

    if (!out)
    {
        std::cerr << "output file error\n" << std::endl;
        return -1;
    }

    out.close();

    return 0;
}


void addState(char *file, fvec u)
{
    std::ofstream out(file,std::ios_base::app);
    for(unsigned int i=0; i < u.n_rows; ++i){
        out << u(i) << " ";
    }
    out << std::endl;
    out.close();
}

using namespace std;


int main (int argc, char* argv[])
{
    CStopWatch timer;
    int dim=2;
    int num=400;
    fmat state(dim,num);
    fvec measurement(dim);
    fvec estimation(dim);
    frowvec startWeights = ones<frowvec>(num);
    startWeights = startWeights / (float) num;
    if (argc<3){
        printf("Error");
        return -1;
    }
    char *file_in= argv[1];
    char *file_out = argv[2];

    std::ifstream in(file_in,std::ios_base::in);	//set input file

    /* initialize particle filter, estimator and resampler */
    BFilterSIR *filter = new BFilterSIR;
    Resampling *resampler = new ResamplingMultinomial;
    Estimation *estimator = new EstimationMean;
    Model *model = new ModelLinear;

    /* specify estimator and resampler that are used for particle filter*/
    filter->setResampling(resampler);
    filter->setEstimation(estimator);
    filter->setModel(model);

    if (!in)
    {
        std::cerr << "input file error\n" << std::endl;
        return -1;
    }

    initStateFile(file_out);	// initialize output file

    timer.startTimer();

    /*read the first measurement and put it to state */
    for (int j=0; j< dim; ++j){
        in >> measurement(j);
        for (int i=0; i<num;++i){
            state(j,i)=measurement(j);
        }
    }
    addState(file_out, measurement);

    filter->setParticles(state,startWeights);
    filter->setThresholdByFactor((float)0.3);
    int counter = 0;
    while (!in.eof())
    {
        std::cout << ++counter << ": \n";
        std::cout << "read measurement ....." << std::endl;

        std::cout << "filter ..." << std::endl;

        filter->predict();

        std::cout << "state predicted" << std::endl;

        for (int j=0; j< dim; ++j){
            in >> measurement(j);
        }

        filter->update(measurement);

        estimation = filter->getEstimation();
        addState(file_out, estimation);

        std::cout << "state written" << std::endl << std::endl;
    }

    timer.stopTimer();

    in.close();

    printf("Done! in %e seconds\n", timer.getElapsedTime());

    delete filter;
    delete estimator;
    delete resampler;

    return 0;
}
