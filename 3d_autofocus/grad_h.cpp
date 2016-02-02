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

void z_vec(const vector<dcomp> phi_offsets, const
        double *Br, const double *Bi, vector<dcomp> &Z, size_t K, size_t
        B_len)
{
    size_t N = B_len / K;

    mxAssert(B_len % K == 0, "length(B) should always be a multiple of K");
    mxAssert(Z.size() == N, "Z must be big enough to fit N values");

    // ------------------------------------------------------------------------
    // Form Z
    // ------------------------------------------------------------------------
    for (size_t n = 0; n < N; ++n) {
        dcomp z_n;

        for (size_t k = 0; k < K; ++k) {
            z_n += dcomp(*Br++, *Bi++) * exp(- J * phi_offsets[k]);
        }

        Z[n] = z_n;
    }
}

// Returns the entropy of the complex image `Z`
double H(vector<dcomp> Z) {
    double Ez = 0, entropy = 0;

    // Returns the total image energy of the complex image Z given the magnitude of
    // the pixels in Z
    for (vector<dcomp>::iterator z = Z.begin(); z != Z.end(); ++z) {
        // TODO: Could be faster to not use `conj.real()` and instead use a*a, b*b
        *z = *z * conj(*z);
        Ez += z->real();
    }

    for (vector<dcomp>::iterator z_mag = Z.begin(); z_mag != Z.end(); ++z_mag) {
        double val = z_mag->real();
        // mxAssert(z_mag->imag() == 0, "");
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

    vector<dcomp> Z(N, 0);

    vector<dcomp> phi_offsets(K, 0);
    for (vector<dcomp>::iterator phi_i = phi_offsets.begin(); phi_i !=
            phi_offsets.end(); ++phi_i)
    {
        *phi_i = dcomp(*phi_offsets_r++, *phi_offsets_i++);
    }

    mexPrintf("In gradH, about to compute Z\n");
    z_vec(phi_offsets, Br, Bi, Z, K, B_len);
    mexPrintf("Computed Z\n");
    double H_not = H(Z);
    mexPrintf("Computed H_not\n");

    for (size_t k = 0; k < K; ++k) {
        if (k > 0) {
            phi_offsets[k - 1].real(phi_offsets[k - 1].real() - delta);
        }

        phi_offsets[k].real(phi_offsets[k].real() + delta);

        z_vec(phi_offsets, Br, Bi, Z, K, B_len);

        grad[k] = (H(Z) - H_not) / delta;
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
