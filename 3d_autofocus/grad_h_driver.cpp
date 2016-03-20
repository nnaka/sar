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
    const size_t K     = 1000;
    const size_t B_len = K * 10 * 10 * 10;

    double *pr = new double[K];
    double *pi = new double[K];
    double *Br = new double[B_len];
    double *Bi = new double[B_len];

    double *grad = new double[K];

    srand(7);

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
        gradH(pi, Br, Bi, grad, K, B_len);

        for (size_t i(0); i < K; ++i) {
            printf("grad[%lu]=%f\n", i, grad[i]);
        }
    }

    delete[] pr;
    delete[] pi;
    delete[] Br;
    delete[] Bi;
    delete[] grad;

    return 0;
}
