// image.cpp
// Defines z_vec(phi), where B is a 1D representation of a image
// Usage:
//        outputVector = image(phi_offset_vector, image_vector)
//

#include <cmath>
#include <complex>
#include <vector>

#include "mex.h"
#include "matrix.h"

using namespace std;

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *phi_offsets_r, *phi_offsets_i, *Br, *Bi, *Zr, *Zi;
    mwIndex K, B_len, N;

    if (nrhs != 2) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs", "Two inputs required.");
    }

    if (nlhs != 1) {
        mexErrMsgIdAndTxt("Autofocus:image:nlhs", "One output required.");
    }

    if (!mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("Autofocus:image:nrhs",
                "'phi_offsets' must be complex.");
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

    // TODO: (joshpfosi) Validate arguments, aggressively

    // Get pointers to real and imaginary parts of input arrays
    phi_offsets_r = mxGetPr(prhs[0]);
    phi_offsets_i = mxGetPi(prhs[0]);
    Br            = mxGetPr(prhs[1]);
    Bi            = mxGetPi(prhs[1]);

    K     = mxGetN(prhs[0]);
    B_len = mxGetN(prhs[1]);

    N = B_len / K;

    plhs[0] = mxCreateDoubleMatrix(1, N, mxCOMPLEX);
    Zr      = mxGetPr(plhs[0]);
    Zi      = mxGetPi(plhs[0]);

    const complex<double> J(0, 1);

    // Form complex vector for easier manipulation
    vector< complex<double> > B(B_len, 0);
    for (mwIndex i = 0; i < B_len; ++i) { // TODO: Use auto
        B[i] = complex<double>(Br[i], Bi[i]);
    }

    // Form complex vector for easier manipulation
    //
    // AND
    //
    // Form 1D array of e^-j * phi_i which repeats every kth element to allow
    // for simple elementwise multiplication on B. See equation (2) in
    // 'tech_report.pdf'.
    //
    // MATLAB: arr = repmat(exp(-1j * phi_offsets), 1, N);
    vector< complex<double> > phi_offsets(K, 0);
    for (mwIndex i = 0; i < K; ++i) { // TODO: Use auto
        complex<double> phi_offset(phi_offsets_r[i], phi_offsets_i[i]);
        phi_offsets[i] = exp(- J * phi_offset);
    }

    // ------------------------------------------------------------------------
    // Form Z
    // ------------------------------------------------------------------------
    for (size_t n = 0; n < N; ++n) {
        complex<double> z_n;

        for (size_t k = 0; k < K; ++k) {
            z_n += B[n * K + k] * phi_offsets[k];
        }

        Zr[n] = z_n.real();
        Zi[n] = z_n.imag();
    }
}
