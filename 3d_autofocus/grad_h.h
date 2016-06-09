#pragma once

#include <cstdlib>

const double delta = 1e-3;

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
        double *grad, std::size_t K, std::size_t B_len,
        double H0, double *Zr, double *Zi);

double H_not(const double *P, const double *Br, const double *Bi,
        double *Zr, double *Zi, size_t K, size_t B_len);
