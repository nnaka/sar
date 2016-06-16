#include "entropy.h"

#include <cmath>
#include "assert.h"

double H(const double *P, const double *Br, const double *Bi,
        double *Zr, double *Zi, size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double acc = 0;

    assert(B_len % K == 0); // length(B) should always be a multiple of K

    double *Z_mag = new double[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    double a, b, c, d;
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

        assert(Z_mag[n] > 0);
        acc += Z_mag[n] * log(Z_mag[n]);
    }

    delete[] Z_mag;
    return -acc;
}
