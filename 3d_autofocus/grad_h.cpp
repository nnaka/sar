#include "grad_h.h"

#if MATLAB_MEX_FILE
#include "mex.h"
#define PRINTF mexPrintf
#else
#include <stdio.h>
#define PRINTF printf
#endif

#include <cmath>
#include <vector>
#include <thread>
#include <assert.h>

using namespace std;

static const auto nthreads = 64 - 1;
static thread threads[nthreads];

inline double entropy(double acc, double Ez)
{
  return (acc - Ez * log(Ez)) / Ez;
}

// Returns the entropy of the complex image specified by `P`, the phase offset
// vector, and `B` the pulse history. Additionally, `H_not` populates `Zr` and
// `Zi` with the resulting image.
double H_not(const double *P, const double *Br, const double *Bi,
        double *Zr, double *Zi, size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0;

    assert(B_len % K == 0); // length(B) should always be a multiple of K

    double *Z_mag = new double[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    double a, b, c, d;
    double acc = 0;
    for (size_t n = 0; n < N; ++n) {
        Zr[n] = 0; Zi[n] = 0;

        for (size_t k(0); k < K; ++k) {
            a = *Br++;
            b = *Bi++;
            sincos(P[k], &d, &c);

            Zr[n] += (a * c + b * d);
            Zi[n] += (b * c - a * d);
        }

        Z_mag[n] = Zr[n] * Zr[n] + Zi[n] * Zi[n];

        // Returns the total image energy of the complex image Z_mag given the
        // magnitude of // the pixels in Z_mag
        Ez += Z_mag[n];
        acc += Z_mag[n] * log(Z_mag[n]);
    }

    delete[] Z_mag;
    return -entropy(acc, Ez);
}

void populate_grad_k(double *grad_i, double H0, const
        double *Br, const double *Bi, const double *Zr, const double *Zi, double
        Ar, double Ai, size_t K, size_t N, size_t k)
{
    double Ez = 0, acc = 0;

    for (size_t n(0); n < N; ++n) {
        double Zn_r = Ar * Br[n * K + k] - Ai * Bi[n * K + k] + Zr[n];
        double Zn_i = Ar * Bi[n * K + k] + Ai * Br[n * K + k] + Zi[n];

        double Zn_mag = Zn_r * Zn_r + Zn_i * Zn_i;

        Ez += Zn_mag;
        acc += Zn_mag * log(Zn_mag);
    }

    *grad_i = (-entropy(acc, Ez) - H0) / delta;
}

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
        double *grad, size_t K, size_t B_len, double H0, double *Zr, double *Zi)
{
    size_t N = B_len / K;
    assert(B_len % K == 0); // length(B) should always be a multiple of K

    double *Ar = new double[K], *Ai = new double[K];

    // Compute alpha
    double sin_phi, cos_phi;
    double sin_delt, cos_delt;
    
    sincos(delta, &sin_delt, &cos_delt);

    for (size_t k(0); k < K; ++k) {
        sincos(phi_offsets[k], &sin_phi, &cos_phi);

        Ar[k] = (-sin_delt) * sin_phi + cos_delt * cos_phi - cos_phi;
        Ai[k] = sin_delt * (-cos_phi) - cos_delt * sin_phi + sin_phi;
    }

    size_t k(0);
    while (k < K) {
        int i;
        for (i = 0; i < nthreads && k < K; ++i) {
            threads[i] = thread(populate_grad_k, grad++, H0,
                        Br, Bi, Zr, Zi, Ar[k], Ai[k], K, N, k);
            ++k;
        }

        // Keep main thread busy but only if we have more to do
        if (k < K) {
            populate_grad_k(grad++, H0, Br, Bi, Zr, Zi, Ar[k], Ai[k], K, N, k);
            ++k;
        }

        while (i > 0) { threads[--i].join(); }
    }

    delete[] Ar;
    delete[] Ai;
}
