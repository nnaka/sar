#pragma once

#include <cstdlib>

// Returns the entropy of the complex image specified by `P`, the phase offset
// vector, and `B` the pulse history. Additionally, `H` populates `Zr` and
// `Zi` with the resulting image.
//
// @param P [K array] array of phase offsets to shift B when computing H
// @param Br [B_len array] array of the real parts of B
// @param Bi [B_len array] array of the imaginary parts of B
// @param Zr [N array] populated with the real parts of the image Z
// @param Zi [N array] populated with the imaginary parts of the image Z
// @param Ez [Scalar] populated with the total energy of Z
// @param acc [Scalar] populated with the summation portion of H
// @param K [Scalar] number of pulses in the pulse history
// @param B_len [Scalar] size of the pulse history (determines N)
double H(const double *P, const double *Br, const double *Bi,
        double *Zr, double *Zi, size_t K, size_t B_len);
