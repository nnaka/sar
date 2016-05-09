// grad_h.cpp
// Computes a finite difference approximation to H given phase offsets and
// complex pulse history
// Usage:
//        grad = grad_h_mex(phi_offset_vector, image_vector)
//

#include "mex.h"
#include "matrix.h"

#include "grad_h.h"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *phi_offsets, *Br, *Bi, *grad;
    size_t K, B_len;

    if (nrhs != 2) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs", "Two inputs required.");
    }

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nlhs", "One output required.");
    }

    if (mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'phi_offsets' must be real.");
    }

    if (mxGetM(prhs[0]) != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'phi_offsets' must be a row vector.");
    }

    if (!mxIsComplex(prhs[1]) || mxGetM(prhs[1]) != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'B' must be complex.");
    }

    if (mxGetM(prhs[1]) != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'B' must be a row vector.");
    }

    // Get pointers to real and imaginary parts of input arrays
    phi_offsets = mxGetPr(prhs[0]);
    Br          = mxGetPr(prhs[1]);
    Bi          = mxGetPi(prhs[1]);

    K     = mxGetN(prhs[0]);
    B_len = mxGetN(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix(1, K, mxREAL);
    grad    = mxGetPr(plhs[0]);

    gradH(phi_offsets, Br, Bi, grad, K, B_len);
}
