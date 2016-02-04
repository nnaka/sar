// grad_h.cpp
// Computes a finite difference approximation to H given phase offsets and
// complex pulse history
// Usage:
//        gradVector = gradH(phi_offset_vector, image_vector)
//

#include "mex.h"
#include "matrix.h"

#include <cmath>

using namespace std;

const double delta = 1;

// Returns the entropy of the complex image `Z`
double H(const double *Pr, const double *Pi, const double *Br, const double *Bi,
        size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0, entropy = 0;

    mxAssert(B_len % K == 0, "length(B) should always be a multiple of K");

    double *Z_mag = new double[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    double z_r, z_i, a, b, c, d;
    for (size_t n = 0; n < N; ++n) {
        z_r = 0; z_i = 0;

        for (size_t k = 0; k < K; ++k) {
            // `b_i * e^-j*phi_i` in rectangular form

            a = *Br++ * exp(Pi[k]);
            b = *Bi++ * exp(Pi[k]);
            c = cos(Pr[k]);
            d = sin(Pr[k]);

            z_r += (a * c + b * d);
            z_i += (b * c - a * d);
        }

        Z_mag[n] = z_r * z_r + z_i * z_i;
    }

    // Returns the total image energy of the complex image Z_mag given the
    // magnitude of // the pixels in Z_mag
    for (size_t n = 0; n < N; ++n) {
        Ez += Z_mag[n];
    }

    double z_intensity = 0;
    for (size_t n = 0; n < N; ++n) {
        z_intensity = Z_mag[n] / Ez;
        entropy += z_intensity * log(z_intensity);
    }

    delete Z_mag;
    return - entropy;
}

// TODO: Nice doc comments
// TODO: Pass references to vectors B, and phi_offsets?
void gradH(double *phi_offsets_r, double *phi_offsets_i, const
        double *Br, const double *Bi, double *grad, size_t K, size_t B_len)
{
    mexPrintf("In gradH, about to compute Z\n");
    mexPrintf("Computed Z\n");
    double H_not = H(phi_offsets_r, phi_offsets_i, Br, Bi, K, B_len);
    mexPrintf("Computed H_not\n");

    for (size_t k = 0; k < K; ++k) {
        if (k > 0) {
            phi_offsets_r[k - 1] -= delta;
        }

        phi_offsets_r[k] += delta;

        double H_i = H(phi_offsets_r, phi_offsets_i, Br, Bi, K, B_len);

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
    grad    = mxGetPr(plhs[0]);

    gradH(phi_offsets_r, phi_offsets_i, Br, Bi, grad, K, B_len);
}
