// ----------------------------------------------------------------------------
// The following is a C++ implementation of the standard gradient descent
// minimization of the image entropy cost function. The below algorithm
// implements the technique described analytically in 'tech_report.pdf'.
//
// @param B [K * N array] pulse history formatted as a 1D array of length `K *
// N` array where `K` refers to the number of pulses in the aperture and `N`
// refers to the number of pixels in either the 2D or 3D image. The array should
// be formatted such that the `i`th set of `K` elements correspond to the
// contributions to pixel `i`.  E.g. the first `K` elements represent each
// pulse's contribution to pixel 0.
//
// @param K [Integer] number of pulses to form `B`
//
// @return focusedImage [Array] X by Y (by Z) focused image
// @return minEntropy [Float] entropy of focused `B`
// @return origEntropy [Float] entropy of unfocused image
// ----------------------------------------------------------------------------
// TODO: `delta` should be a parameter to gradFunc

#include "mex.h"
#include "matrix.h"

#include <assert.h>

#include "minimize_entropy.h"

#define B_ARG 0
#define K_ARG 1
#define FOC_IMAGE 0
#define MIN_ENTR  1
#define ORIG_ENTR 2

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *focusedImageR, *focusedImageI, *minEntropy, *origEntropy;

    float *Br, *Bi;
    size_t K, B_len, N;

    if (nrhs != 2) {
        mexErrMsgIdAndTxt("minimizeEntropy:nrhs", "Two inputs required.");
    }

    if (nlhs != 3) {
        mexErrMsgIdAndTxt("minimizeEntropy:nlhs", "Three output required.");
    }

    // TODO: Validate `K`

    if (!mxIsComplex(prhs[B_ARG]) || mxGetM(prhs[B_ARG]) != 1 ||
            !mxIsSingle(prhs[B_ARG])) {
        mexErrMsgIdAndTxt("minimizeEntropy:nrhs",
                "'B' must be complex and single precision.");
    }

    // Assign to each input
    // NOTE: We cast to float * with the assumption that the underlying data
    // types are actually floats and not doubles
    Br = (float *)mxGetData(prhs[B_ARG]);
    Bi = (float *)mxGetImagData(prhs[B_ARG]);
    K     = mxGetScalar(prhs[K_ARG]);
    B_len = mxGetN(prhs[B_ARG]);

    mxAssert(B_len % K == 0, "`B_len` must be a multiple of `K`");

    N = B_len / K;

    // Allocate return arguments
    plhs[FOC_IMAGE] = mxCreateNumericMatrix(1, N, mxSINGLE_CLASS, mxCOMPLEX);
    plhs[MIN_ENTR]  = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
    plhs[ORIG_ENTR] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);

    if (plhs[FOC_IMAGE] == NULL) {
        mexErrMsgIdAndTxt("minimizeEntropy:plhs",
                "Cannot allocate sufficient memory");
    }

    // Assign to each output
    // NOTE: We cast to float * with the assumption that the underlying data
    // types are actually floats and not doubles
    focusedImageR = (float *)mxGetData(plhs[FOC_IMAGE]);
    focusedImageI = (float *)mxGetImagData(plhs[FOC_IMAGE]);

    minEntropy    = (float *)mxGetData(plhs[MIN_ENTR]);
    origEntropy   = (float *)mxGetData(plhs[ORIG_ENTR]);

    minimize_entropy(focusedImageR, focusedImageI, minEntropy, origEntropy,
            Br, Bi, K, N);
}
