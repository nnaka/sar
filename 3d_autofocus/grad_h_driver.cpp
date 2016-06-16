// Driver program for 'gradH' for profiling

#include "grad_h.h"

#include <stdio.h>
#include <string.h> // memset
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

double drand() {
    return (double)rand() / (double)rand();
}

int main() {
    const size_t K     = 100;
    const size_t N     = 501 * 501;
    const size_t B_len = K * N;

    double *P = new double[K];
    double *Br = new double[B_len];
    double *Bi = new double[B_len];

    double *Zr = new double[N];
    double *Zi = new double[N];

    double *grad = new double[K];

    srand(7);

    for (size_t n = 0; n < N; ++n) {
        Zr[n] = drand();
        Zi[n] = drand();
    }

    for (size_t k = 0; k < K; ++k) {
        P[k] = drand();
    }

    for (size_t b = 0; b < B_len; ++b) {
        Br[b] = drand();
        Bi[b] = drand();
    }

    for (int i = 0; i < 1; ++i) {
        printf("Iteration %d\n", i);
        gradH(P, Br, Bi, grad, K, B_len, 2, Zr, Zi);

        for (size_t i(K - 10); i < K; ++i) {
            printf("grad[%lu]=%f\n", i, grad[i]);
        }
    }

    delete[] P;
    delete[] Br;
    delete[] Bi;
    delete[] Zr;
    delete[] Zi;
    delete[] grad;

    return 0;
}
