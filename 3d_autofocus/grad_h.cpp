// grad_h.cpp
// Computes a finite difference approximation to H given phase offsets and
// complex pulse history
// Usage:
//        gradVector = gradH(phi_offset_vector, image_vector)
//

#include "mex.h"
#include "matrix.h"

#include "entropy.h"

#include <cmath>
#include <vector>

using namespace std;

// Returns the entropy of the complex image `Z`
double H(const vector<dcomp> &phi_offsets, const double *Br, const double *Bi,
        size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0, entropy = 0;

    mxAssert(B_len % K == 0, "length(B) should always be a multiple of K");

    vector<dcomp> Z_mag(N);

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    for (size_t n = 0; n < N; ++n) {
        dcomp z_n;

        for (size_t k = 0; k < K; ++k) {
            z_n += dcomp(*Br++, *Bi++) * exp(- J * phi_offsets[k]);
        }

        // TODO: Could be faster to not use `conj.real()` and instead use a*a, b*b
        Z_mag[n] = z_n * conj(z_n);
    }

    // Returns the total image energy of the complex image Z_mag given the magnitude of
    // the pixels in Z_mag
    for (vector<dcomp>::iterator z = Z_mag.begin(); z != Z_mag.end(); ++z) {
        Ez += z->real();
    }

    for (vector<dcomp>::iterator z_mag = Z_mag.begin(); z_mag != Z_mag.end(); ++z_mag) {
        double val = z_mag->real();
        val /= Ez;
        entropy += val * log(val);
    }

    return - entropy;
}

// TODO: Nice doc comments
// TODO: Pass references to vectors B, and phi_offsets?
void gradH(double *phi_offsets_r, double *phi_offsets_i, const
        double *Br, const double *Bi, double *grad, size_t K, size_t B_len)
{
    mxAssert(B_len % K == 0, "length(B) should always be a multiple of K");
    size_t N = B_len / K;

    const double delta = 1;

    vector<dcomp> phi_offsets(K);
    for (vector<dcomp>::iterator phi_i = phi_offsets.begin(); phi_i !=
            phi_offsets.end(); ++phi_i)
    {
        *phi_i = dcomp(*phi_offsets_r++, *phi_offsets_i++);
    }

    mexPrintf("In gradH, about to compute Z\n");
    mexPrintf("Computed Z\n");
    double H_not = H(phi_offsets, Br, Bi, K, B_len);
    mexPrintf("Computed H_not\n");

    for (size_t k = 0; k < K; ++k) {
        if (k > 0) {
            phi_offsets[k - 1].real(phi_offsets[k - 1].real() - delta);
        }

        phi_offsets[k].real(phi_offsets[k].real() + delta);

        double H_i = H(phi_offsets, Br, Bi, K, B_len);

        grad[k] = (H_i - H_not) / delta;
    }
}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *phi_offsets_r, *phi_offsets_i, *Br, *Bi, *grad;
    size_t K, B_len, N;

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

    // Get pointers to real and imaginary parts of input arrays
    phi_offsets_r = mxGetPr(prhs[0]);
    phi_offsets_i = mxGetPi(prhs[0]);
    Br            = mxGetPr(prhs[1]);
    Bi            = mxGetPi(prhs[1]);

    K     = mxGetN(prhs[0]);
    B_len = mxGetN(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix(1, K, mxREAL);
    grad      = mxGetPr(plhs[0]);

    gradH(phi_offsets_r, phi_offsets_i, Br, Bi, grad, K, B_len);
}
