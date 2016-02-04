// Driver program for 'gradH' for profiling

#include <stdio.h>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

// Returns the entropy of the complex image `Z`
double H(const double *Pr, const double *Pi, const double *Br, const double *Bi,
        size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0, entropy = 0;

    double *Z_mag = (double *)malloc(sizeof(*Z_mag) * N);

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

    free(Z_mag);
    return - entropy;
}

// TODO: Nice doc comments
// TODO: Pass references to vectors B, and phi_offsets?
void gradH(double *phi_offsets_r, double *phi_offsets_i, const
        double *Br, const double *Bi, double *grad, size_t K, size_t B_len)
{
    const double delta = 1;

    double H_not = H(phi_offsets_r, phi_offsets_i, Br, Bi, K, B_len);

    for (size_t k = 0; k < K; ++k) {
        if (k > 0) {
            phi_offsets_r[k - 1] -= delta;
        }

        phi_offsets_r[k] += delta;

        double H_i = H(phi_offsets_r, phi_offsets_i, Br, Bi, K, B_len);

        grad[k] = (H_i - H_not) / delta;
    }
}


double drand() {
    return (double)rand() / (double)rand();
}

int main() {
    const size_t K     = 900;
    const size_t B_len = K * 50 * 50 * 50;

    double *pr = new double[K];
    double *pi = new double[K];
    double *Br = new double[B_len];
    double *Bi = new double[B_len];

    double *grad = new double[K];

    srand(time(0));

    for (size_t k = 0; k < K; ++k) {
        pr[k] = drand();
        pi[k] = drand();
    }

    for (size_t b = 0; b < B_len; ++b) {
        Br[b] = drand();
        Bi[b] = drand();
    }

    for (int i = 0; i < 1; ++i) {
        printf("Iteration %d\n", i);
        gradH(pi, pr, Br, Bi, grad, K, B_len);
    }

    free(pr);
    free(pi);
    free(Br);
    free(Bi);
    free(grad);

    return 0;
}
