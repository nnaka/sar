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

inline float entropy(double acc, double Ez)
{
  return (acc - Ez * log(Ez)) / Ez;
}

void populate_grad_k(float *grad_i, float H0, const
        float *Br, const float *Bi, const float *Zr, const float *Zi, float
        Ar, float Ai, size_t K, size_t N, size_t k)
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
void gradH(float *phi_offsets, const float *Br, const float *Bi,
        float *grad, size_t K, size_t B_len, float H0, float *Zr, float *Zi)
{
    size_t N = B_len / K;
    assert(B_len % K == 0); // length(B) should always be a multiple of K

    float *Ar = new float[K], *Ai = new float[K];

    // Compute alpha
    float sin_phi, cos_phi;
    float sin_delt, cos_delt;
    
    sincosf(delta, &sin_delt, &cos_delt);

    for (size_t k(0); k < K; ++k) {
        sincosf(phi_offsets[k], &sin_phi, &cos_phi);

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
