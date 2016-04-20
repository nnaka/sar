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
#include <cublas_v2.h>

using namespace std;

const int nT = 1024;

#define checkCuda(result) checkCudaInner(result, __LINE__)

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCudaInner(cudaError_t result, int lineno)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s on line %d\n", cudaGetErrorString(result), lineno);
    assert(result == cudaSuccess);
  }
  return result;
}

__host__ __device__ inline double entropy(double acc, double Ez)
{
  assert(Ez > 0);
  return (acc - Ez * log(Ez)) / Ez;
}

// B is a matrix with K columns and N rows
template<typename T>
__global__ void computeEz(const T * __restrict__ Br, const T * __restrict__ Bi, 
    const T * __restrict__ Zr, const T * __restrict__ Zi,
    const T * __restrict__ Ar, const T * __restrict__ Ai,
    T * __restrict__ g_Ez, T * __restrict__ g_acc, size_t N, size_t K)
{
  extern __shared__ T sdata[];

  T *Ez = sdata, *acc = &sdata[blockDim.x];

  T x(0.0), y(0.0);

  const int k = blockIdx.x;

  // Accumulate per thread partial sum over columns of B*
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    double Zn_r = Ar[k] * Br[n * K + k] - Ai[k] * Bi[n * K + k] + Zr[n];
    double Zn_i = Ar[k] * Bi[n * K + k] + Ai[k] * Br[n * K + k] + Zi[n];

    double Zn_mag = Zn_r * Zn_r + Zn_i * Zn_i;

    assert(Zn_mag >= 0);
    x += Zn_mag;
    y += Zn_mag * log(Zn_mag);
  }

  // load thread partial sum into shared memory
  Ez[threadIdx.x] = x;
  acc[threadIdx.x] = y;

  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      Ez[threadIdx.x] += Ez[threadIdx.x + offset];
      acc[threadIdx.x] += acc[threadIdx.x + offset];
    }
    __syncthreads();
  }

  // thread 0 writes the final result
  if (threadIdx.x == 0) {
    g_Ez[blockIdx.x] += Ez[0];
    g_acc[blockIdx.x] += acc[0];
  }
}

// B is a matrix with K columns and N rows
template<typename T>
__global__ void kernelSum(const T * __restrict__ Br, const T * __restrict__ Bi, 
    T * __restrict__ Zr, T * __restrict__ Zi,
    const size_t K, T * __restrict__ Z_mag, const T * __restrict__ P)
{
  extern __shared__ T sdata[];

  T *s1 = sdata, *s2 = &sdata[blockDim.x];

  T x(0.0), y(0.0);

  const T * Br_row = &Br[blockIdx.x * K];
  const T * Bi_row = &Bi[blockIdx.x * K];

  // Accumulate per thread partial sum
  double sin, cos;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    sincos(P[i % K], &sin, &cos);
    x += Br_row[i] * cos + Bi_row[i] * sin;
    y += Bi_row[i] * cos - Br_row[i] * sin;
  }

  // load thread partial sum into shared memory
  s1[threadIdx.x] = x;
  s2[threadIdx.x] = y;

  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s1[threadIdx.x] += s1[threadIdx.x + offset];
      s2[threadIdx.x] += s2[threadIdx.x + offset];
    }
    __syncthreads();
  }

  // thread 0 writes the final result
  if (threadIdx.x == 0) {
    Zr[blockIdx.x] = s1[0];
    Zi[blockIdx.x] = s2[0];

    Z_mag[blockIdx.x] = s1[0] * s1[0] + s2[0] * s2[0];
    assert(Z_mag[blockIdx.x] >= 0);
  }
}

template <class T>
__global__ void computeEntropy(T *Z_mag, T *d_acc, unsigned int n)
{
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  assert(Z_mag[i] >= 0);
  sdata[tid] = (i < n) ? Z_mag[i] * log(Z_mag[i]) : 0;

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=1; s < blockDim.x; s *= 2)
  {
    // modulo arithmetic is slow!
    if ((tid % (2*s)) == 0)
    {
      sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) {
    d_acc[blockIdx.x] = sdata[0];
  }
}

__global__ void computeAlpha(const double *P, double sin_delt, double cos_delt,
    double *Ar, double *Ai, size_t K)
{
  unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

  double sin_phi, cos_phi;
  if (k < K) {
    sincos(P[k], &sin_phi, &cos_phi);

    Ar[k] = (-sin_delt) * sin_phi + cos_delt * cos_phi - cos_phi;
    Ai[k] = sin_delt * (-cos_phi) - cos_delt * sin_phi + sin_phi;
  }
}

__global__ void computeGrad(double *grad, const double *acc, const double *Ez, double H0, double delta, size_t K)
{
  unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

  if (k < K) {
    grad[k] = (-entropy(acc[k], Ez[k]) - H0) / delta;
  }
}

// Returns the entropy of the complex image `Z`
void H_not(const double *d_P, double *d_Br, double *d_Bi, double *Zr, double
    *Zi, double *Ez, double *acc, size_t K, size_t B_len)
{
  const size_t N = B_len / K;

  assert(B_len % K == 0); // length(B) should always be a multiple of K

  double *d_Z_mag = NULL;
  checkCuda(cudaMalloc((void **)&d_Z_mag, N * sizeof(double)));

  kernelSum<double><<<N, nT, 2 * nT * sizeof(double)>>>(d_Br, d_Bi, Zr, Zi, K, d_Z_mag, d_P);

  int bs = (N + nT - 1) / nT; // cheap ceil()

  double *d_accum = NULL;
  double *accum = NULL;

  checkCuda(cudaMallocHost((void **)&accum, bs * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_accum, bs * sizeof(double)));

  cublasHandle_t handle;

  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

  // cublasDasum sums the absolute values, but Z_mag is always positive so this
  // is correct
  double sum = 0;
  cublasDasum(handle, N, d_Z_mag, 1, &sum);
  *Ez += sum;

  computeEntropy<double><<<bs, nT, nT * sizeof(double)>>>(d_Z_mag, d_accum, N);
  checkCuda(cudaMemcpy(accum, d_accum, bs * sizeof(double), cudaMemcpyDeviceToHost));
  for (size_t b(0); b < bs; ++b) { *acc += accum[b]; }

  checkCuda(cudaFree(d_Z_mag));
  checkCuda(cudaFree(d_accum));
  cublasDestroy(handle);
}

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
    double *grad, size_t K, size_t B_len)
{
  // const auto maxMem = 2147483648 / 2; // 1 GB - size of half of GPU physical memory
  const size_t N = B_len / K;

  // Solve for N:
  // (4 * K + 2 * N + 2 * K * N_prime) * sizeof(double);
  // size_t N_prime = min((maxMem / sizeof(double) - 4 * K - 2 * N) / (2 * K), N);
  size_t N_prime = 64000;
  // TODO: Determine a more precise value
  size_t B_len_prime = N_prime * K;

  double *d_Br, *d_Bi, *d_P, *d_Zr, *d_Zi, *d_Ez, *d_acc, *d_Ar, *d_Ai, *d_grad;
  double Ez = 0, acc = 0;

  // TODO: Use pinned memory
  checkCuda(cudaMalloc((void **)&d_P,  K * sizeof(double)));
  checkCuda(cudaMemcpy(d_P, phi_offsets, K * sizeof(double), cudaMemcpyHostToDevice));

  checkCuda(cudaMalloc((void **)&d_Zr, N * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Zi, N * sizeof(double)));

  checkCuda(cudaMalloc((void **)&d_Br, B_len_prime * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Bi, B_len_prime * sizeof(double)));

  checkCuda(cudaMalloc((void **)&d_Ar,  K * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Ai,  K * sizeof(double)));

  checkCuda(cudaMalloc((void **)&d_Ez, K * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_acc, K * sizeof(double)));

  checkCuda(cudaMalloc((void **)&d_grad, K * sizeof(double)));

  checkCuda(cudaMemset(d_Ez, 0, K * sizeof(double)));
  checkCuda(cudaMemset(d_acc, 0, K * sizeof(double)));

  double sin_delt, cos_delt;
  sincos(delta, &sin_delt, &cos_delt);
  computeAlpha<<<(K + nT - 1) / nT, nT>>>(d_P, sin_delt, cos_delt, d_Ar, d_Ai, K);

  PRINTF("In gradH, about to compute Z\n");
  PRINTF("Computed Z\n");

  size_t num_iter = ceil((float)B_len / B_len_prime);

  for (size_t i(0); i < num_iter; ++i)
  {
    size_t len = min(B_len_prime, B_len - i * B_len_prime);

    checkCuda(cudaMemcpy(d_Br, &Br[i * B_len_prime], len * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_Bi, &Bi[i * B_len_prime], len * sizeof(double), cudaMemcpyHostToDevice));

    H_not(d_P, d_Br, d_Bi, &d_Zr[i * N_prime], &d_Zi[i * N_prime], &Ez, &acc, K, len);
    computeEz<double><<<K, nT, 2 * nT * sizeof(double)>>>(d_Br, d_Bi, &d_Zr[i * N_prime], &d_Zi[i * N_prime], d_Ar, d_Ai, d_Ez, d_acc, len / K, K);
  }

  double H0 = -entropy(acc, Ez);
  PRINTF("Computed H_not=%f\n", H0);

  computeGrad<<<(K + nT - 1) / nT, nT>>>(d_grad, d_acc, d_Ez, H0, delta, K);
  checkCuda(cudaMemcpy(grad, d_grad, K * sizeof(double), cudaMemcpyDeviceToHost));

  checkCuda(cudaFree(d_Br));
  checkCuda(cudaFree(d_Bi));

  checkCuda(cudaFree(d_Zr));
  checkCuda(cudaFree(d_Zi));

  checkCuda(cudaFree(d_P));

  checkCuda(cudaFree(d_Ar));
  checkCuda(cudaFree(d_Ai));

  checkCuda(cudaFree(d_grad));
}
