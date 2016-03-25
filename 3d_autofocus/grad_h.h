#pragma once

#include <cstdlib>

const double delta = 1e-3;

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
        double *grad, std::size_t K, std::size_t B_len);
