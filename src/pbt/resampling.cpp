#include "resampling.h"
#include <stdlib.h>


Particles Resampling::resample(Particles* particles){
	Particles resampledSet = *particles;

	return resampledSet;
}

std::vector<int> Resampling::getAssignmentVec()
{
    return assignmentVec;
}

Resampling::Resampling(){
    
}

Resampling::~Resampling(){

}
