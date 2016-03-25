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

__device__ inline double entropy(double acc, double Ez)
{
  return (acc - Ez * log(Ez)) / Ez;
}

// B is a matrix with K columns and N rows
template<typename T>
__global__ void computeGrad(const T * __restrict__ Br, const T * __restrict__ Bi, 
    const T * __restrict__ Zr, const T * __restrict__ Zi,
    const T * __restrict__ Ar, const T * __restrict__ Ai,
    T * __restrict__ grad, size_t N, size_t K, double H0, double delta)
{
  extern __shared__ T sdata[];

  T *Ez = sdata, *acc = &sdata[blockDim.x];

  T x(0.0), y(0.0);

  const T * Br_col = &Br[blockIdx.x];
  const T * Bi_col = &Bi[blockIdx.x];

  // Accumulate per thread partial sum
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    int col_idx = i * K;

    double Zn_r = Ar[blockIdx.x] * Br_col[col_idx] - Ai[blockIdx.x] * Bi_col[col_idx] + Zr[i];
    double Zn_i = Ar[blockIdx.x] * Bi_col[col_idx] + Ai[blockIdx.x] * Br_col[col_idx] + Zi[i];

    double Zn_mag = Zn_r * Zn_r + Zn_i * Zn_i;

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
    grad[blockIdx.x] = (-entropy(acc[0], Ez[0]) - H0) / delta;
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
  }
}

template <class T>
__global__ void computeEntropy(T *g_idata, T *g_odata, double *Ez, unsigned int n)
{
  extern __shared__ T sdata[];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? (g_idata[i] / *Ez) * log(g_idata[i] / *Ez) : 0;

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
    g_odata[blockIdx.x] = sdata[0];
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

// Returns the entropy of the complex image `Z`
double H_not(const double *d_P, double *d_Br, double *d_Bi, double *Zr, double
    *Zi, size_t K, size_t B_len, cudaStream_t *stream, int nStreams, int
    streamSize)
{
  size_t N = B_len / K;
  double entropy(0);

  assert(B_len % K == 0); // length(B) should always be a multiple of K

  double *d_Z_mag = NULL;

  checkCuda(cudaMalloc((void **)&d_Z_mag, N * sizeof(double)));

  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    int Z_offset = i * (N / nStreams);

    kernelSum<double><<<N / nStreams, nT, 2 * nT * sizeof(double),
      stream[i]>>>(&d_Br[offset], &d_Bi[offset], &Zr[Z_offset], &Zi[Z_offset], K, &d_Z_mag[Z_offset], d_P);
  }

  cudaDeviceSynchronize();

  int bs = (N + nT - 1) / nT; // cheap ceil()

  double *d_Ez = NULL;
  double *d_accum = NULL;
  double *accum = NULL;

  checkCuda(cudaMallocHost((void **)&accum, bs * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_accum, bs * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Ez, sizeof(double)));

  cublasHandle_t handle;

  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  // cublasDasum sums the absolute values, but Z_mag is always positive so this
  // is correct
  cublasDasum(handle, N, d_Z_mag, 1, d_Ez);

  computeEntropy<double><<<bs, nT, nT * sizeof(double)>>>(d_Z_mag, d_accum, d_Ez, N);

  checkCuda(cudaMemcpy(accum, d_accum, bs * sizeof(double), cudaMemcpyDeviceToHost));

  for (size_t b(0); b < bs; ++b) { entropy += accum[b]; }

  checkCuda(cudaFree(d_Z_mag));
  checkCuda(cudaFree(d_accum));
  checkCuda(cudaFree(d_Ez));
  cublasDestroy(handle);

  return -entropy;
}

void gradH_priv(double *d_P, const double *d_Br, const double *d_Bi,
    double *d_grad, const double *d_Zr, const double *d_Zi, size_t K, size_t B_len, double H0)
{
  const size_t N = B_len / K;

  double *d_Ar, *d_Ai;

  checkCuda(cudaMalloc((void **)&d_Ar,  K * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Ai,  K * sizeof(double)));

  double sin_delt, cos_delt;

  sincos(delta, &sin_delt, &cos_delt);

  int bs = (K + nT - 1) / nT; // cheap ceil
  computeAlpha<<<bs, nT, 0>>>(d_P, sin_delt, cos_delt, d_Ar, d_Ai, K);

  computeGrad<double><<<K, nT, 2 * nT * sizeof(double)>>>(d_Br, d_Bi, d_Zr, d_Zi, d_Ar, d_Ai, d_grad, N, K, H0, delta);

  checkCuda(cudaFree(d_Ai));
  checkCuda(cudaFree(d_Ar));
}

// TODO: Nice doc comments
void gradH(double *phi_offsets, const double *Br, const double *Bi,
    double *grad, size_t K, size_t B_len)
{
  // const int maxMem = 2147483648; // 2 GB
  const int N = B_len / K;

  // Solve for K:
  // (4 * K + 2 * N + 2 * K * N) * sizeof(double);
  // int K_prime = min(((maxMem / sizeof(double)) - 2 * N) / (4 + 2 * N), K);

  const int nStreams = (B_len > 8) ? 8 : 1;
  const int streamSize = B_len / nStreams;

  cudaStream_t stream[nStreams];

  for (int i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamCreate(&stream[i]));

  double *d_Br, *d_Bi, *d_P, *d_Zr, *d_Zi, *d_grad;

  // TODO: Use pinned memory
  checkCuda(cudaMalloc((void **)&d_P,  K * sizeof(double)));

  checkCuda(cudaMemcpy(d_P, phi_offsets, K * sizeof(double), cudaMemcpyHostToDevice));

  // TODO: Use pinned memory
  checkCuda(cudaMalloc((void **)&d_Br, B_len * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Bi, B_len * sizeof(double)));

  checkCuda(cudaMalloc((void **)&d_Zr, N * sizeof(double)));
  checkCuda(cudaMalloc((void **)&d_Zi, N * sizeof(double)));

  checkCuda(cudaMalloc((void **)&d_grad, K * sizeof(double)));

  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;

    checkCuda(cudaMemcpyAsync(&d_Br[offset], &Br[offset], streamSize * sizeof(double), cudaMemcpyHostToDevice, stream[i]));
    checkCuda(cudaMemcpyAsync(&d_Bi[offset], &Bi[offset], streamSize * sizeof(double), cudaMemcpyHostToDevice, stream[i]));
  }

  PRINTF("In gradH, about to compute Z\n");
  PRINTF("Computed Z\n");
  double H0 = H_not(d_P, d_Br, d_Bi, d_Zr, d_Zi, K, B_len, stream, nStreams, streamSize);
  PRINTF("Computed H_not\n");

  gradH_priv(d_P, d_Br, d_Bi, d_grad, d_Zr, d_Zi, K, B_len, H0);

  checkCuda(cudaMemcpy(grad, d_grad, K * sizeof(double), cudaMemcpyDeviceToHost));

  checkCuda(cudaFree(d_Br));
  checkCuda(cudaFree(d_Bi));

  checkCuda(cudaFree(d_Zr));
  checkCuda(cudaFree(d_Zi));

  checkCuda(cudaFree(d_P));

  for (int i = 0; i < nStreams; ++i)
    checkCuda(cudaStreamDestroy(stream[i]));
}
