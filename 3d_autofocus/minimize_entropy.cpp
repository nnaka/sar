#include "minimize_entropy.h"

#if MATLAB_MEX_FILE
#include "mex.h"
#define PRINTF mexPrintf
#else
#include <stdio.h>
#define PRINTF printf
#endif

#include "grad_h.h"
#include "entropy.h"
#include <vector>
#include <cmath>

using namespace std;

inline double entropy(double acc, double Ez)
{
  return (acc - Ez * log(Ez)) / Ez;
}

void minimize_entropy(double *focusedImageR, double *focusedImageI, double
        *minEntropy, double *origEntropy, double *Br, double *Bi, size_t K,
        size_t N)
{
    // Step size parameter for gradient descent
    auto s                       = 100;
    // Difference after which iteration "converges"
    auto const convergenceThresh = 0.001;
    // Minimum step size
    auto const stepMinimum       = 0.01;
    auto const maxIter           = 100;

    auto l = 1; // First iteration is all 0s, so start at iteration 1

    // Holds array of potentially minimizing phase offsets (guessing zero
    // initially). 50 is an arbitrary guess for the number of iterations
    vector< vector<double> > phiOffsets(maxIter, vector<double>(K, 0));

    *minEntropy = H(&(phiOffsets[0])[0], Br, Bi,
            focusedImageR, focusedImageI, K, N * K);
    *origEntropy = *minEntropy;

    double *grad = new double[K];

    double *tempImageR = new double[N], *tempImageI = new double[N];
    double tempEntropy = *minEntropy;

    while (l < maxIter) {
        // Compute `grad`
        gradH(&(phiOffsets[l - 1])[0], Br, Bi, grad, K, N * K,
                tempEntropy, focusedImageR, focusedImageI);

        // Compute new phase offsets after descending down gradient
        for (size_t k = 0; k < K; ++k) {
            phiOffsets[l][k] = phiOffsets[l - 1][k] - s * grad[k];
        }

        tempEntropy = H(&(phiOffsets[l])[0], Br, Bi,
                tempImageR, tempImageI, K, N * K);

        PRINTF("tempEntropy = %f, minEntropy = %f\n", tempEntropy, *minEntropy);

        if (*minEntropy < tempEntropy) {
            s /= 2;

            PRINTF("Reducing step size to %d\n", s);

            if (s < stepMinimum) {
                PRINTF("s is below minimum so breaking\n");
                break;
            }
        }
        else {
            if (*minEntropy - tempEntropy < convergenceThresh) {
                PRINTF("Change in entropy (%f - %f = %f) < %f\n",
                        *minEntropy, tempEntropy, *minEntropy - tempEntropy,
                        convergenceThresh);
                break; // if decreases in entropy are small
            }

            *minEntropy = tempEntropy;

            for (size_t n(0); n < N; ++n) {
                focusedImageR[n] = tempImageR[n];
                focusedImageI[n] = tempImageI[n];
            }

            ++l;
        }
    }

    delete[] tempImageR;
    delete[] tempImageI;
    delete[] grad;
}

