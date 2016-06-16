#include "grad_h.h"

#include <cmath>
#include <vector>
#include <thread>
#include <assert.h>

using namespace std;

static const auto nthreads = 64 - 1;
static thread threads[nthreads];

// TODO: Nice doc comments
void populate_grad_k(double *grad_i, const double *Br, const double *Bi, const
        double *Zr, const double *Zi,
        double phi_i, size_t K, size_t N, size_t i)
{
    double sin_phi, cos_phi;
    sincos(phi_i, &sin_phi, &cos_phi);

    for (size_t n(0); n < N; ++n) {
        double Z_mag = Zr[n] * Zr[n] + Zi[n] * Zi[n];
        double derZ_mag = 4 *
            (Br[n * K + i] * cos_phi + Bi[n * K + i] * sin_phi) * 
            (Bi[n * K + i] * cos_phi - Br[n * K + i] * sin_phi);

        assert(Z_mag > 0);
        *grad_i += derZ_mag * (1 + log(Z_mag));
    }
}

// TODO: Nice doc comments
void gradH(double *P, const double *Br, const double *Bi,
        double *grad, size_t K, size_t B_len, double *Zr, double *Zi)
{
    size_t N = B_len / K;
    assert(B_len % K == 0); // length(B) should always be a multiple of K

    size_t k(0);
    while (k < K) {
        int i;
        for (i = 0; i < nthreads && k < K; ++i) {
            threads[i] = thread(populate_grad_k, grad++,
                        Br, Bi, Zr, Zi, P[k], K, N, k);
            ++k;
        }

        // Keep main thread busy but only if we have more to do
        if (k < K) {
            populate_grad_k(grad++, Br, Bi, Zr, Zi, P[k], K, N, k);
            ++k;
        }

        while (i > 0) { threads[--i].join(); }
    }
}
