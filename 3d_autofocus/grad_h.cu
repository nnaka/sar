#include "grad_h.h"

#if MATLAB_MEX_FILE
#include "mex.h"
#define PRINTF mexPrintf
#else
#define PRINTF printf
#endif

#include <cmath>
#include <vector>
#include <thread>
#include <assert.h>

#include <cuda_runtime.h>

using namespace std;

__global__ void reduce0(double *P, double *z, double *Br, double *Bi, int K, bool is_z_r) {
  extern __shared__ double sdata[];

  double *p_r = sdata;

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (is_z_r) {
    p_r[tid] = Br[i] * cos(P[i]) + Bi[i] * sin(P[i]);
  }
  else {
    p_r[tid] = Bi[i] * cos(P[i]) - Br[i] * sin(P[i]);
  }

  __syncthreads();

  // do reduction in shared mem
  for(unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      p_r[tid] += p_r[tid + s];
    }

    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    z[blockIdx.x] = p_r[0];
  }
}

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
        double *grad, size_t K, size_t B_len)
{
    vector<double> P(phi_offsets, phi_offsets + K);

    PRINTF("In gradH, about to compute Z\n");
    PRINTF("Computed Z\n");
    double H_not = H(P, Br, Bi, K, B_len);
    PRINTF("Computed H_not\n");

    auto Pr_k(P.begin());

    while (Pr_k != P.end()) {
      if (Pr_k != P.begin()) {
        *(Pr_k - 1) -= delta;
      }

      *Pr_k++ += delta;

      populate_grad_k(grad++, H_not, P, Br, Bi, K, B_len);
    }
}

#define funcCheck(stmt) {                                            \
  cudaError_t err = stmt;                                          \
  if (err != cudaSuccess)                                          \
  {                                                                \
    printf( "Failed to run stmt %d\n", __LINE__);                 \
    printf( "Got CUDA error ...  %s\n", cudaGetErrorString(err)); \
    return -1;                                                   \
  }                                                                \
}

// Returns the entropy of the complex image `Z`
double H(const vector<double> P, const double *Br, const double *Bi,
        size_t K, size_t B_len)
{
    size_t N = B_len / K;
    double Ez = 0, entropy = 0;

    assert(B_len % K == 0); // length(B) should always be a multiple of K

    double *Z_mag = new double[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    const int bs = 4; // NOTE: Must be a multiple of K
    const int nT = K / bs;

    double *z_r = (double *)malloc(bs * sizeof(double));
    double *z_i = (double *)malloc(bs * sizeof(double));

    double *d_P, *d_Br, *d_Bi;
    double *d_z_r = NULL, *d_z_i = NULL;

    funcCheck(cudaMalloc((void **)&d_P,  K * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_Br, B_len * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_Bi, B_len * sizeof(double)));

    funcCheck(cudaMalloc((void **)&d_z_r, bs * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_z_i, bs * sizeof(double)));

    funcCheck(cudaMemcpy(d_P, &P[0], K * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_Br, Br, B_len * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_Bi, Bi, B_len * sizeof(double), cudaMemcpyHostToDevice));

    for (size_t n = 0; n < N; ++n) {
      reduce0<<<bs, nT, nT * sizeof(double)>>>(d_P, d_z_r, d_Br + n * K, d_Bi + n * K, K, true);
      reduce0<<<bs, nT, nT * sizeof(double)>>>(d_P, d_z_i, d_Br + n * K, d_Bi + n * K, K, false);

      funcCheck(cudaMemcpy(z_r, d_z_r, bs * sizeof(double), cudaMemcpyDeviceToHost));
      funcCheck(cudaMemcpy(z_i, d_z_i, bs * sizeof(double), cudaMemcpyDeviceToHost));

      for (size_t b(1); b < bs; ++b) { z_r[0] += z_r[b]; }
      for (size_t b(1); b < bs; ++b) { z_i[0] += z_i[b]; }

      Z_mag[n] = *z_r * *z_r + *z_i * *z_i;
    }

    // Returns the total image energy of the complex image Z_mag given the
    // magnitude of // the pixels in Z_mag
    for (size_t n = 0; n < N; ++n) { Ez += Z_mag[n]; }

    funcCheck(cudaFree(d_P));
    funcCheck(cudaFree(d_Br));
    funcCheck(cudaFree(d_Bi));

    funcCheck(cudaFree(d_z_r));
    funcCheck(cudaFree(d_z_i));

    free(z_r);
    free(z_i);

    double z_intensity = 0;
    for (size_t n = 0; n < N; ++n) {
        z_intensity = Z_mag[n] / Ez;
        entropy += z_intensity * log(z_intensity);
    }

    delete[] Z_mag;
    return - entropy;
}

void populate_grad_k(double *grad_i, double H_not, const vector<double> P,
        const double *Br, const double *Bi, size_t K,
        size_t B_len)
{
    double H_i = H(P, Br, Bi, K, B_len);
    *grad_i = (H_i - H_not) / delta;
}

