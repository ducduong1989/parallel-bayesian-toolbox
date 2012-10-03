/*#include "estimation_modema.h"
#include <vector>
#include <iostream>

Particle EstimationModema::getEstimation(std::vector<Particle> input, int numberOfParticles){
    float sum=0.0;
    float weightSum = 0.0;
    const int dim = input[0].dim;
    Particle  estimation(2);

    for (int j=0;j<dim;++j){
            sum = 0.0;
            weightSum=0.0;
            for (int i=0; i<input.size(); ++i){
                    sum += input[i].value[j] * input[i].weight;
                    weightSum += input[i].weight;
            }
            estimation.value[j] = sum / weightSum;
    }
    return estimation;
}

EstimationModema::~EstimationModema(){

}

EstimationModema::EstimationModema(){

}*/
