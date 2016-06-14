#pragma once 

#include <cstdlib>

void minimize_entropy(float *focusedImageR, float *focusedImageI, float
        *minEntropy, float *origEntropy, float *Br, float *Bi, size_t K,
        size_t N);
