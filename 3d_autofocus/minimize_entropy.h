#pragma once 

#include <cstdlib>

void minimize_entropy(double *focusedImageR, double *focusedImageI, double
        *minEntropy, double *origEntropy, double *Br, double *Bi, size_t K,
        size_t N);
