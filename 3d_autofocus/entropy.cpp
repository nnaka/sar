#include "entropy.h"

#include <cmath>
#include "assert.h"

inline float entropy(float acc, float Ez)
{
  return (acc - Ez * log(Ez)) / Ez;
}

float H(const float *P, const float *Br, const float *Bi,
        float *Zr, float *Zi, size_t K, size_t B_len)
{
    size_t N = B_len / K;
    float Ez = 0;

    assert(B_len % K == 0); // length(B) should always be a multiple of K

    float *Z_mag = new float[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    float a, b, c, d;
    float acc = 0;
    for (size_t n = 0; n < N; ++n) {
        Zr[n] = 0; Zi[n] = 0;

        for (size_t k(0); k < K; ++k) {
            a = *Br++;
            b = *Bi++;
            sincosf(P[k], &d, &c);

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
