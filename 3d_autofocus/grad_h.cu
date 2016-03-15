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

__global__ void reduce0(double *P, double *Br_shift, double *Bi_shift, double *Br, double *Bi, size_t K, size_t B_len)
{
  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < B_len) {
    Br_shift[n] = Br[n] * cos(P[n % K]) + Bi[n] * sin(P[n % K]);
    Bi_shift[n] = Bi[n] * cos(P[n % K]) - Br[n] * sin(P[n % K]);
  }
}

__global__ void reduce1(double *Z_mag, double *z_r, double *z_i, size_t N)
{
  unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < N) {
    Z_mag[n] = z_r[n] * z_r[n] + z_i[n] * z_i[n];
  }
}

__global__ void sum(double *in, double *out) {
  extern __shared__ double sdata[];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = in[i];
  __syncthreads();

  // do reduction in shared mem
  for(unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    out[blockIdx.x] = sdata[0];
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
    double Ez(0), entropy(0);

    assert(B_len % K == 0); // length(B) should always be a multiple of K

    double *Z_mag = new double[N];

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    double *z_r = new double[N], *z_i = new double[N];
    memset(z_r, 0.0, sizeof(double) * N); memset(z_i, 0.0, sizeof(double) * N);

    double *Br_shift = new double[B_len], *Bi_shift = new double[B_len];
    memset(Br_shift, 0.0, sizeof(double) * B_len); memset(Bi_shift, 0.0, sizeof(double) * B_len);

    const int nT = 1024;
    const int bs = (B_len + nT - 1) / nT; // cheap ceil()

    double *d_P, *d_Br, *d_Bi;
    double *d_Br_shift = NULL, *d_Bi_shift = NULL;
    double *d_z_r = NULL, *d_z_i = NULL;
    double *d_Z_mag = NULL;

    funcCheck(cudaMalloc((void **)&d_P,  K * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_Br, B_len * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_Bi, B_len * sizeof(double)));

    funcCheck(cudaMalloc((void **)&d_Br_shift, B_len * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_Bi_shift, B_len * sizeof(double)));

    funcCheck(cudaMalloc((void **)&d_z_r, N * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_z_i, N * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_Z_mag, N * sizeof(double)));

    funcCheck(cudaMemset(d_Br_shift, 0, B_len * sizeof(double)));
    funcCheck(cudaMemset(d_Bi_shift, 0, B_len * sizeof(double)));

    funcCheck(cudaMemset(d_Z_mag, 0, N * sizeof(double)));

    funcCheck(cudaMemcpy(d_P, &P[0], K * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_Br, Br, B_len * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_Bi, Bi, B_len * sizeof(double), cudaMemcpyHostToDevice));

    reduce0<<<bs, nT>>>(d_P, d_Br_shift, d_Bi_shift, d_Br, d_Bi, K, B_len);

    funcCheck(cudaMemcpy(Br_shift, d_Br_shift, B_len * sizeof(double), cudaMemcpyDeviceToHost));
    funcCheck(cudaMemcpy(Bi_shift, d_Bi_shift, B_len * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t n = 0; n < B_len; ++n) {
      z_r[n / K] += Br_shift[n];
      z_i[n / K] += Bi_shift[n];
    }

    funcCheck(cudaMemcpy(d_z_r, z_r, N * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_z_i, z_i, N * sizeof(double), cudaMemcpyHostToDevice));

    reduce1<<<bs, nT>>>(d_Z_mag, d_z_r, d_z_i, N);

    funcCheck(cudaMemcpy(Z_mag, d_Z_mag, N * sizeof(double), cudaMemcpyDeviceToHost));

    delete[] z_r; delete[] z_i;
    delete[] Br_shift; delete[] Bi_shift;

    funcCheck(cudaFree(d_P));
    funcCheck(cudaFree(d_Br));
    funcCheck(cudaFree(d_Bi));

    funcCheck(cudaFree(d_Br_shift));
    funcCheck(cudaFree(d_Bi_shift));

    funcCheck(cudaFree(d_z_r));
    funcCheck(cudaFree(d_z_i));
    funcCheck(cudaFree(d_Z_mag));

    // Returns the total image energy of the complex image Z_mag given the
    // magnitude of // the pixels in Z_mag
    for (size_t n = 0; n < N; ++n) { Ez += Z_mag[n]; }

    double z_intensity(0);
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

