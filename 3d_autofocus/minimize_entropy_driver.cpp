// Driver program for 'minimizeEntropy' for profiling

#include "minimize_entropy.h"

#include <stdio.h>
#include <string.h> // memset
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

double drand() {
    return (double)rand() / (double)rand();
}

int main() {
    const size_t K = 50;
    const size_t N = 501 * 501;
    const size_t B_len = N * K;

    double *Br = new double[B_len];
    double *Bi = new double[B_len];

    double *focusedImageR = new double[N];
    double *focusedImageI = new double[N];
    double minEntropy = 0, origEntropy = 0;

    srand(7);

    for (size_t b = 0; b < B_len; ++b) {
        Br[b] = drand();
        Bi[b] = drand();
    }

    for (int i = 0; i < 1; ++i) {
        printf("Iteration %d\n", i);
        minimize_entropy(focusedImageR, focusedImageI, &minEntropy,
                &origEntropy, Br, Bi, K, N);
    }

    delete[] Br;
    delete[] Bi;
    delete[] focusedImageR;
    delete[] focusedImageI;

    return 0;
}
