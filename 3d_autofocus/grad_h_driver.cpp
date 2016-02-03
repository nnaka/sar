// Driver program for 'gradH' for profiling

#include "entropy.h"

#include <stdio.h>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

// Returns the entropy of the complex image `Z`
double H(const vector<dcomp> &phi_offsets, const double *Br, const double *Bi,
        size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0, entropy = 0;


    vector<dcomp> Z_mag(N);

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    for (size_t n = 0; n < N; ++n) {
        dcomp z_n;

        for (size_t k = 0; k < K; ++k) {
            z_n += dcomp(*Br++, *Bi++) * exp(- J * phi_offsets[k]);
        }

        // TODO: Could be faster to not use `conj.real()` and instead use a*a, b*b
        Z_mag[n] = z_n * conj(z_n);
    }

    // Returns the total image energy of the complex image Z_mag given the magnitude of
    // the pixels in Z_mag
    for (vector<dcomp>::iterator z = Z_mag.begin(); z != Z_mag.end(); ++z) {
        Ez += z->real();
    }

    for (vector<dcomp>::iterator z_mag = Z_mag.begin(); z_mag != Z_mag.end(); ++z_mag) {
        double val = z_mag->real();
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
    const double delta = 1;

    vector<dcomp> phi_offsets(K);
    for (vector<dcomp>::iterator phi_i = phi_offsets.begin(); phi_i !=
            phi_offsets.end(); ++phi_i)
    {
        *phi_i = dcomp(*phi_offsets_r++, *phi_offsets_i++);
    }

    double H_not = H(phi_offsets, Br, Bi, K, B_len);

    for (size_t k = 0; k < K; ++k) {
        if (k > 0) {
            phi_offsets[k - 1].real(phi_offsets[k - 1].real() - delta);
        }

        phi_offsets[k].real(phi_offsets[k].real() + delta);

        double H_i = H(phi_offsets, Br, Bi, K, B_len);

        grad[k] = (H_i - H_not) / delta;
    }
}


double drand() {
    return (double)rand() / (double)rand();
}

int main() {
    const size_t K     = 400;
    const size_t B_len = 400 * 5 * 5 * 5;

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

    for (int i = 0; i < 3; ++i) {
        printf("Status: i=%d\n", i);
        gradH(pi, pr, Br, Bi, grad, K, B_len);
    }

    free(pr);
    free(pi);
    free(Br);
    free(Bi);
    free(grad);

    return 0;
}
