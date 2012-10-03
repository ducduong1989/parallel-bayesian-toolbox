#include <fstream>
#include <string>
#include <math.h>
#include <time.h>
#include "model.h"
#include <stdlib.h>
#include <armadillo>
#include "model_sinus.h"

using namespace arma;

int initStateFile(char *file)
{
        std::ofstream out(file,std::ios_base::trunc);

        if (!out)
        {
            perror("output file error\n");
                return -1;
        }

        out.close();

        return 0;
}


void addState(char *file, fmat u)
{
        std::ofstream out(file,std::ios_base::app);
        for(unsigned int i=0; i < u.n_rows; ++i){
                out << u[i] << " ";
        }
        out << std::endl;
        out.close();
}


int main (int argc, char* argv[]){
        int num = 0;
        if (argc<3){
                perror("Error");
                printf("Please call program like:\n");
                printf("./datagen filename_original filename_measurement numberOfStates\n");
                return -1;
        }
        char *file_original = argv[1];
        char *file_out = argv[2];
        num = atoi(argv[3]);

        Model *process = new ModelSinus();
        process->initialize();

        printf("Dimension of state: %d\n",process->getProcessDimension());

        fmat state(process->getProcessDimension(),1);

        fmat measurement(process->getMeasurementDimension(),1);

        state.zeros();

        initStateFile(file_out);	// initialize output file
        initStateFile(file_original);

        addState(file_out, state);


        for (int k=0; k<num;k++)
        {
            state = process->ffun(&state);
            printf("x: %e  y: %e\n",state(0),state(1));
            addState(file_original,state);
            measurement = process->hfun(&state);
            printf("x: %e  y: %e\n\n",measurement(0),measurement(1));
            addState(file_out, measurement);
        }

        printf("Done!");

        return 0;
}

