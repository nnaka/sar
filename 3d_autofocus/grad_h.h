#pragma once

#include <cstdlib>

const double delta = 1e-3;

// Computes the gradient of the entropy function, H, by finite difference
// approximation
//
// @param P [K array] array of phase offsets to shift B when computing H
// @param Br [B_len array] array of the real parts of B
// @param Bi [B_len array] array of the imaginary parts of B
// @param grad [K array] populated with gradient of H at `phi_offsets`
// @param H0 [Scalar] entropy of image shifted by `phi_offsets`
// @param K [Scalar] number of pulses in the pulse history
// @param B_len [Scalar] size of the pulse history (determines N)
// @param Zr [N array] populated with the real parts of the image Z
// @param Zi [N array] populated with the imaginary parts of the image Z
void gradH(double *phi_offsets, const double *Br, const double *Bi,
        double *grad, std::size_t K, std::size_t B_len,
        double H0, double *Zr, double *Zi);
