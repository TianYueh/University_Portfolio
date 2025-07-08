#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <stdint.h>

void mexFunction(int nlhs, mxArray* plhs[],int nrhs, const mxArray* prhs[]){
    if(nrhs != 2){
        mexErrMsgTxt("Two input args required for c input.");
    }
    if(nlhs != 1){
        mexErrMsgTxt("One output arg required for c output.");
    }
    if(!mxIsDouble(prhs[0])||mxIsComplex(prhs[0])){
        mexErrMsgTxt("First input must be real double array.");
    }
    if(!mxIsDouble(prhs[1])||mxIsComplex(prhs[1])){
        mexErrMsgTxt("Second input must be real double array.");
    }
    size_t nA = mxGetNumberOfElements(prhs[0]);
    double* A = mxGetPr(prhs[0]);
    size_t nB_rows = mxGetM(prhs[1]);
    size_t nB_cols = mxGetN(prhs[1]);
    if(nB_cols != 2){
        mexErrMsgTxt("Second input must have 2 cols.");
    }
    double* B = mxGetPr(prhs[1]);
    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), mxGetDimensions(prhs[0]), mxINT32_CLASS, mxREAL);
    int32_t* C = (int32_t*)mxGetData(plhs[0]);

    for(size_t k = 0;k<nA;k++){
        C[k] = 0;
        for(size_t q = 0;q<nB_rows;q++){
            if(A[k]>=B[q] && A[k]<B[q+nB_rows]){
                C[k] = (int32_t)(q+1);
                break;
            }
        }
    }
}