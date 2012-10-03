#include "mex.h"
#include "math.h"
#include<armadillo>

using namespace arma;

class MatlabInterface {
public:

	void matlab2arma(mat& A, const mxArray *mxdata);

	void freeVar(mat& A, const double *ptr);

	mat floatToDoubleArmaMat(fmat);

	fmat DoubleToFloatArmaMat(mat);

};
