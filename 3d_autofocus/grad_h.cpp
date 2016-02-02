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

void z_vec(const double *phi_offsets_r, const double *phi_offsets_i, const
        double *Br, const double *Bi, double *Zr, double *Zi, size_t
        K, size_t B_len)
{
    // TODO: Assert B_len % K == 0
    size_t N = B_len / K;

    // Form complex vector for easier manipulation
    //
    // AND
    //
    // Form 1D array of e^-j * phi_i which repeats every kth element to allow
    // for simple elementwise multiplication on B. See equation (2) in
    // 'tech_report.pdf'.
    //
    // MATLAB: arr = repmat(exp(-1j * phi_offsets), 1, N);
    vector<dcomp> phi_offsets(K, 0);
    for (size_t i = 0; i < K; ++i) {
        dcomp phi_offset(phi_offsets_r[i], phi_offsets_i[i]);
        phi_offsets[i] = exp(- J * phi_offset);
    }

    // ------------------------------------------------------------------------
    // Form Z
    // ------------------------------------------------------------------------
    size_t i = 0;
    for (size_t n = 0; n < N; ++n) {
        dcomp z_n;

        for (size_t k = 0; k < K; ++k) {
            z_n += dcomp(Br[i], Bi[i]) * phi_offsets[k];
            ++i;
        }

        Zr[n] = z_n.real();
        Zi[n] = z_n.imag();
    }
}

// Returns the entropy of the complex image `Z`
// TODO: This function is GROSSLY inefficient but correct
double H(const double *Zr, const double *Zi, size_t N) {
    double Ez = 0, entropy = 0;

    vector<dcomp> Z(N, 0); vector<double> Z_mag(N, 0);

    for (size_t n = 0; n < N; ++n) {
        Z[n] = dcomp(Zr[n], Zi[n]);
    }

    // Returns the total image energy of the complex image Z given the magnitude of
    // the pixels in Z
    for (size_t i = 0; i < N; ++i) {
        Z_mag[i] = (Z[i] * conj(Z[i])).real();
    }

    for (vector<double>::iterator it = Z_mag.begin(); it != Z_mag.end(); ++it) {
        Ez += *it;
    }

    for (vector<double>::iterator it = Z_mag.begin(); it != Z_mag.end(); ++it) {
        *it /= Ez;
    }

    for (vector<double>::iterator it = Z_mag.begin(); it != Z_mag.end(); ++it) {
        entropy += *it * log(*it);
    }

    return - entropy;
}

// TODO: Nice doc comments
// TODO: Pass references to vectors B, and phi_offsets?
void gradH(double *phi_offsets_r, double *phi_offsets_i, const
        double *Br, const double *Bi, double *grad, size_t K, size_t B_len)
{
    // TODO: Assert B_len % K == 0
    size_t N = B_len / K;

    const double delta = 1;

    double Zr[N], Zi[N];

    mexPrintf("In gradH, about to compute Z\n");
    z_vec(phi_offsets_r, phi_offsets_i, Br, Bi, Zr, Zi, K, B_len);
    mexPrintf("Computed Z\n");
    double H_not = H(Zr, Zi, N);
    mexPrintf("Computed H_not\n");

    for (size_t k = 0; k < K; ++k) {
        if (k > 0) {
            phi_offsets_r[k - 1] -= delta;
        }

        phi_offsets_r[k] += delta;

        z_vec(phi_offsets_r, phi_offsets_i, Br, Bi, Zr, Zi, K, B_len);

        grad[k] = (H(Zr, Zi, N) - H_not) / delta;
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
