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

template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

template <class T>
__global__ void sum(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

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
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, double Ez, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? (g_idata[i] / Ez) * log(g_idata[i] / Ez) : 0;

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
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
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

    // ------------------------------------------------------------------------
    // Form Z_mag
    // ------------------------------------------------------------------------
    double *Z_mag    = NULL;
    double *z_r      = NULL;
    double *z_i      = NULL;
    double *Br_shift = NULL;
    double *Bi_shift = NULL;

    funcCheck(cudaMallocHost((void **)&Z_mag, N * sizeof(double)));
    funcCheck(cudaMallocHost((void **)&z_r, N * sizeof(double)));
    funcCheck(cudaMallocHost((void **)&z_i, N * sizeof(double)));
    funcCheck(cudaMallocHost((void **)&Br_shift, B_len * sizeof(double)));
    funcCheck(cudaMallocHost((void **)&Bi_shift, B_len * sizeof(double)));

    const int nT = 1024;
    int bs = (B_len + nT - 1) / nT; // cheap ceil()

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

    funcCheck(cudaMemcpy(d_P, &P[0], K * sizeof(double), cudaMemcpyHostToDevice));

    funcCheck(cudaMemcpy(d_Br, Br, B_len * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_Bi, Bi, B_len * sizeof(double), cudaMemcpyHostToDevice));

    reduce0<<<bs, nT, 0>>>(d_P, d_Br_shift, d_Bi_shift, d_Br, d_Bi, K, B_len);

    funcCheck(cudaMemcpy(Br_shift, d_Br_shift, B_len * sizeof(double), cudaMemcpyDeviceToHost));
    funcCheck(cudaMemcpy(Bi_shift, d_Bi_shift, B_len * sizeof(double), cudaMemcpyDeviceToHost));

    // TODO: Explore reduction kernel
    for (size_t n = 0; n < B_len; ++n) {
      z_r[n / K] += Br_shift[n];
      z_i[n / K] += Bi_shift[n];
    }

    funcCheck(cudaMemcpy(d_z_r, z_r, N * sizeof(double), cudaMemcpyHostToDevice));
    funcCheck(cudaMemcpy(d_z_i, z_i, N * sizeof(double), cudaMemcpyHostToDevice));

    reduce1<<<bs, nT>>>(d_Z_mag, d_z_r, d_z_i, N);

    funcCheck(cudaFree(d_P));
    funcCheck(cudaFree(d_Br));
    funcCheck(cudaFree(d_Bi));

    funcCheck(cudaFree(d_Br_shift));
    funcCheck(cudaFree(d_Bi_shift));

    funcCheck(cudaFree(d_z_r));
    funcCheck(cudaFree(d_z_i));

    // Returns the total image energy of the complex image Z_mag given the
    // magnitude of // the pixels in Z_mag
    // TODO: Explore reduction kernel
    bs = (N + nT - 1) / nT; // cheap ceil()

    double *accum   = NULL;
    double *d_accum = NULL;

    funcCheck(cudaMallocHost((void **)&accum, bs * sizeof(double)));
    funcCheck(cudaMalloc((void **)&d_accum, bs * sizeof(double)));

    sum<double><<<bs, nT, nT * sizeof(double)>>>(d_Z_mag, d_accum, N);
    funcCheck(cudaMemcpy(accum, d_accum, bs * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t b(0); b < bs; ++b) { Ez += accum[b]; }


    reduce2<double><<<bs, nT, nT * sizeof(double)>>>(d_Z_mag, d_accum, Ez, N);
    funcCheck(cudaMemcpy(accum, d_accum, bs * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t b(0); b < bs; ++b) { entropy += accum[b]; }

    funcCheck(cudaFree(d_Z_mag));
    funcCheck(cudaFree(d_accum));

    funcCheck(cudaFreeHost(accum));
    funcCheck(cudaFreeHost(Z_mag));
    funcCheck(cudaFreeHost(z_r));
    funcCheck(cudaFreeHost(z_i));
    funcCheck(cudaFreeHost(Br_shift));
    funcCheck(cudaFreeHost(Bi_shift));

    return - entropy;
}

void populate_grad_k(double *grad_i, double H_not, const vector<double> P,
        const double *Br, const double *Bi, size_t K,
        size_t B_len)
{
    double H_i = H(P, Br, Bi, K, B_len);
    *grad_i = (H_i - H_not) / delta;
}

