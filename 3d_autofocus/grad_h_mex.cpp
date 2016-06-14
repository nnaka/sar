// grad_h.cpp
// Computes a finite difference approximation to H given phase offsets and
// complex pulse history
// Usage:
//        grad = grad_h_mex(phi_offset_vector, image_vector)
//

#include "mex.h"
#include "matrix.h"

#include "grad_h.h"

#define PHI_OFFSETS_ARG 0
#define B_ARG           1

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *phi_offsets, *Br, *Bi, *grad;
    size_t K, B_len;

    if (nrhs != 2) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs", "Two inputs required.");
    }

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nlhs", "One output required.");
    }

    if (mxIsComplex(prhs[PHI_OFFSETS_ARG]) ||
            !mxIsSingle(prhs[PHI_OFFSETS_ARG])) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'phi_offsets' must be real and single precision.");
    }

    if (mxGetM(prhs[PHI_OFFSETS_ARG]) != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'phi_offsets' must be a row vector.");
    }

    if (!mxIsComplex(prhs[B_ARG]) || mxGetM(prhs[B_ARG]) != 1 ||
            !mxIsSingle(prhs[B_ARG])) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'B' must be complex and single precision.");
    }

    if (mxGetM(prhs[B_ARG]) != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'B' must be a row vector.");
    }

    // Get pointers to real and imaginary parts of input arrays
    phi_offsets = (float *)mxGetData(prhs[PHI_OFFSETS_ARG]);
    Br          = (float *)mxGetData(prhs[B_ARG]);
    Bi          = (float *)mxGetImagData(prhs[B_ARG]);

    K     = mxGetN(prhs[PHI_OFFSETS_ARG]);
    B_len = mxGetN(prhs[B_ARG]);

    plhs[0] = mxCreateNumericMatrix(1, K, mxSINGLE_CLASS, mxREAL);
    grad    = (float *)mxGetData(plhs[0]);

    gradH(phi_offsets, Br, Bi, grad, K, B_len);
}
