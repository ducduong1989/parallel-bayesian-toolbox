#include "matlab_interface.h"

using namespace arma;

void MatlabInterface::matlab2arma(mat& A, const mxArray *mxdata){
	// delete [] A.mem; // don't do this!
	access::rw(A.mem) = mxGetPr(mxdata);
	access::rw(A.n_rows) = (uword)mxGetM(mxdata); // transposed!
	access::rw(A.n_cols) = (uword)mxGetN(mxdata);
	access::rw(A.n_elem) = A.n_rows*A.n_cols;
}

void MatlabInterface::freeVar(mat& A, const double *ptr){
	access::rw(A.mem) = ptr;
	access::rw(A.n_rows) = 1; // transposed!
	access::rw(A.n_cols) = 1;
	access::rw(A.n_elem) = 1;
}

mat MatlabInterface::floatToDoubleArmaMat(fmat input){
	mat output(input.n_rows,input.n_cols);
	for (unsigned int i=0;i<input.n_rows;++i){
		for (unsigned int j=0; j<input.n_cols;++j){
			output(i,j)=input(i,j);
		}
	}
	return output;
}

fmat MatlabInterface::DoubleToFloatArmaMat(mat input){
	fmat output(input.n_rows,input.n_cols);
	for (unsigned int i=0;i<input.n_rows;++i){
		for (unsigned int j=0; j<input.n_cols;++j){
			output(i,j)=(float)input(i,j);
		}
	}
	return output;
}

