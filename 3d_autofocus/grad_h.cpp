#include "grad_h.h"

#if MATLAB_MEX_FILE
#include "mex.h"
#define PRINTF mexPrintf
#else
#define PRINTF printf
#endif

#include <cmath>
#include <vector>
#include <thread>
#include <assert.h>

using namespace std;

static const auto nthreads = 64 - 1;
static thread threads[nthreads];

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
        double *grad, size_t K, size_t B_len)
{
    vector<double> P(phi_offsets, phi_offsets + K);

    PRINTF("In gradH, about to compute Z\n");
    PRINTF("Computed Z\n");
    double H_not = H(P, Br, Bi, K, B_len);
    PRINTF("Computed H_not\n");

    auto Pr_k(P.begin());
    while (Pr_k != P.end()) {
        int i;
        for (i = 0; i < nthreads && Pr_k != P.end(); ++i) {
            if (Pr_k != P.begin()) {
                *(Pr_k - 1) -= delta;
            }

            *Pr_k++ += delta;

            threads[i] = thread(populate_grad_k, grad++, H_not,
                        P, Br, Bi, K, B_len);
        }

        // Keep main thread busy but only if we have more to do
        if (Pr_k != P.end()) {
            *(Pr_k - 1) -= delta;
            *Pr_k++ += delta;
            populate_grad_k(grad++, H_not, P, Br, Bi, K, B_len);
        }

        while (i > 0) { threads[--i].join(); }
    }
}

// Returns the entropy of the complex image `Z`
double H(const vector<double> P, const double *Br, const double *Bi,
        size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0, entropy = 0;

    assert(B_len % K == 0); // length(B) should always be a multiple of K

    double *Z_mag = new double[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    double z_r, z_i, a, b, c, d;
    for (size_t n = 0; n < N; ++n) {
        z_r = 0; z_i = 0;

        for (auto pr : P) {
            a = *Br++;
            b = *Bi++;
            c = cos(pr);
            d = sin(pr);

            z_r += (a * c + b * d);
            z_i += (b * c - a * d);
        }

        Z_mag[n] = z_r * z_r + z_i * z_i;

        // Returns the total image energy of the complex image Z_mag given the
        // magnitude of // the pixels in Z_mag
        Ez += Z_mag[n];
    }

    double z_intensity = 0;
    for (size_t n = 0; n < N; ++n) {
        z_intensity = Z_mag[n] / Ez;
        entropy += z_intensity * log(z_intensity);
    }

    delete[] Z_mag;
    return - entropy;
}

void populate_grad_k(double *grad_i, double H_not, const vector<double> P,
        const double *Br, const double *Bi, size_t K,
        size_t B_len)
{
    double H_i = H(P, Br, Bi, K, B_len);
    *grad_i = (H_i - H_not) / delta;
}

