// Driver program for 'minimizeEntropy' for profiling

#include "minimize_entropy.h"

#include <stdio.h>
#include <string.h> // memset
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

float drand() {
    return (float)rand() / (float)rand();
}

int main() {
    const size_t K = 50;
    const size_t N = 501 * 501;
    const size_t B_len = N * K;

    float *Br = new float[B_len];
    float *Bi = new float[B_len];

    float *focusedImageR = new float[N];
    float *focusedImageI = new float[N];
    float minEntropy = 0, origEntropy = 0;

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
