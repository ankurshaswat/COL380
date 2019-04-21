#include "lab3_cuda.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <malloc.h>

using namespace std;

#define LINEAR_BLOCKSIZE 1024
#define SQUARE_BLOCKSIZE 32

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001

bool *changed;
int *ind, *state, N;
double *S, *E, *e, c, s;

__host__ int inline INDEX_HOST(int i, int j, int m, int n) { return i * n + j; }

__host__ int maxind(int k) {
  int m = k + 1, i;

  double max_ = fabs(S[INDEX_HOST(k, m, N, N)]), temp;
  for (i = k + 2; i < N; i++) {
    temp = fabs(S[INDEX_HOST(k, i, N, N)]);
    if (temp > max_) {
      m = i;
      max_ = temp;
    }
  }

  return m;
}

__host__ void update(int k, double t) {
  double ek_prev = e[k];
  e[k] = ek_prev + t;

  if (e[k] < 0)
    e[k] = 0;

  if (changed[k] && fabs(ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
    changed[k] = false;
    *state = *state - 1;
  } else if ((!changed[k]) && fabs(ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
    changed[k] = true;
    *state = *state + 1;
  }
}

__host__ void rotate(int k, int l, int i, int j) {
  double Skl = S[INDEX_HOST(k, l, N, N)], Sij = S[INDEX_HOST(i, j, N, N)];
  S[INDEX_HOST(k, l, N, N)] = c * Skl - s * Sij;
  S[INDEX_HOST(i, j, N, N)] = s * Skl + c * Sij;
}

__host__ void update_e(int k, int l, int i) {
  double Eik = E[INDEX_HOST(i, k, N, N)], Eil = E[INDEX_HOST(i, l, N, N)];
  E[INDEX_HOST(i, k, N, N)] = c * Eik - s * Eil;
  E[INDEX_HOST(i, l, N, N)] = s * Eik + c * Eil;
}

__host__ void JACOBI(int n, double *dev_E, double *dev_e, double *dev_S) {

  N = n;

  E = (double *)malloc(sizeof(double) * N * N);
  e = (double *)malloc(sizeof(double) * N);
  ind = (int *)malloc(sizeof(int) * N);
  changed = (bool *)malloc(sizeof(bool) * N);
  S = (double *)malloc(sizeof(double) * N * N);
  state = (int *)malloc(sizeof(int));

  cudaMemcpy(S, dev_S, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int i, j, m, k, l;
  double p, y, d, r, t, max_, temp;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      E[INDEX_HOST(i, j, N, N)] = 0;
    }
    E[INDEX_HOST(i, i, N, N)] = 1;
  }

  *state = N;
  int count = 0;

  for (k = 0; k < N; k++) {
    ind[k] = maxind(k);
    e[k] = S[INDEX_HOST(k, k, N, N)];
    changed[k] = true;
  }

  while (*state != 0) {
    count++;
    m = 0;

    max_ = fabs(S[INDEX_HOST(m, ind[m], N, N)]);
    for (k = 1; k < N - 1; k++) {
      temp = fabs(S[INDEX_HOST(k, ind[k], N, N)]);
      if (temp > max_) {
        m = k;
        max_ = temp;
      }
    }

    k = m;
    l = ind[m];
    p = S[INDEX_HOST(k, l, N, N)];
    y = (e[l] - e[k]) / 2.0;
    d = fabs(y) + sqrt(p * p + y * y);
    r = sqrt(p * p + d * d);
    c = d / r;
    s = p / r;
    t = (p * p) / d;

    if (y < 0.0) {
      s = -s;
      t = -t;
    }

    S[INDEX_HOST(k, l, N, N)] = 0.0;
    update(k, -t);
    update(l, t);

    for (i = 0; i < k; i++) {
      rotate(i, k, i, l);
    }
    for (i = k + 1; i < l; i++) {
      rotate(k, i, i, l);
    }
    for (i = l + 1; i < N; i++) {
      rotate(k, i, l, i);
    }

    for (i = 0; i < N; i++) {
      update_e(k, l, i);
    }

    ind[k] = maxind(k);
    ind[l] = maxind(l);

    if (count % 1000 == 0) {
      printf("%d %d\n", *state, count);
    }
  }

  cudaMemcpy(dev_E, E, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_e, e, sizeof(double) * N, cudaMemcpyHostToDevice);

  free(changed);
  free(ind);
  free(S);
  free(state);

  cudaDeviceSynchronize();

  free(E);
  free(e);
}

__device__ inline int INDEX(int i1, int i2, int l1, int l2) {
  return i1 * l2 + i2;
}

__global__ void ODD_EVEN_SORT(double *arr, int *indices, int n,
                              bool *converged) {
  int index_global = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  *converged = false;
  bool odd_iter = false;
  double temp;
  int to_see, to_see_next, index_local, i, temp_int;

  for (i = index_global; i < n; i += stride) {
    indices[i] = i;
  }

  while (!(*converged)) {
    __syncthreads();
    *converged = true;
    for (index_local = index_global; index_local < n / 2;
         index_local += stride) {
      if (odd_iter && 2 * index_local + 2 < n) {
        to_see = 2 * index_local + 1;
        to_see_next = 2 * index_local + 2;
        if (arr[to_see] < arr[to_see_next]) {

          temp = arr[to_see_next];
          arr[to_see_next] = arr[to_see];
          arr[to_see] = temp;

          temp_int = indices[to_see_next];
          indices[to_see_next] = indices[to_see];
          indices[to_see] = temp_int;

          *converged = false;
        }
      } else if (!odd_iter && 2 * index_local + 1 < n) {
        to_see = 2 * index_local;
        to_see_next = 2 * index_local + 1;
        if (arr[to_see] < arr[to_see_next]) {

          temp = arr[to_see_next];
          arr[to_see_next] = arr[to_see];
          arr[to_see] = temp;

          temp_int = indices[to_see_next];
          indices[to_see_next] = indices[to_see];
          indices[to_see] = temp_int;

          *converged = false;
        }
      }
    }

    odd_iter = !odd_iter;
  }
}

__global__ void TRANSPOSE(double *M, int m, int n, double *M_T) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n * m) {
    M_T[i] = M[INDEX(i % m, i / m, m, n)];
  }
}

__global__ void MATMUL2(int p, int q, int r, double *A, double *B, double *C) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  double sum = 0;

  if (row < p && col < r) {
    for (i = 0; i < q; i++) {
      sum += A[INDEX(row, i, p, q)] * B[INDEX(i, col, q, r)];
    }
    C[INDEX(row, col, p, r)] = sum;
  }
}

__global__ void ARRANGE(int *indices, double *old_E, double *new_E, int n1,
                        int n2) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n1 * n2) {
    new_E[i] = old_E[INDEX(i / n2, indices[i % n2], n1, n2)];
  }
}

__global__ void GET_SINGULAR_VALS(int n, double *e, double *SIGMA,
                                  double *SIGMA_INV) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double sqrt_;

  if (i < n) {
    sqrt_ = sqrt(e[i]);
    SIGMA[i] = sqrt_;
    SIGMA_INV[i] = 1 / sqrt_;
  }
}

// TODO
__global__ void GET_EIGEN_SUM(double *eigen_total, double *e, int n) {
  int i;
  *eigen_total = 0;
  for (i = 0; i < n; i++) {
    *eigen_total += e[i];
  }
}

__global__ void MULTIPLY_SIGMA_INV(int m, int n, double *M, double *V,
                                   double *SIGMA_INV, double *U) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int i;
  double sum = 0;

  if (row < m && col < m) {
    if (col < n) {
      for (i = 0; i < n; i++) {
        sum += M[INDEX(row, i, m, n)] * V[INDEX(i, col, n, n)];
      }
      U[INDEX(row, col, m, m)] = sum * SIGMA_INV[col];
    } else {
      U[INDEX(row, col, m, m)] = 0;
    }
  }
}

__host__ void GET_U(int m, int n, double *dev_M, double *dev_V,
                    double *dev_SIGMA_INV, double *dev_U) {
  dim3 dimBlock(SQUARE_BLOCKSIZE, SQUARE_BLOCKSIZE);
  dim3 dimGrid((m + SQUARE_BLOCKSIZE - 1) / SQUARE_BLOCKSIZE,
               (m + SQUARE_BLOCKSIZE - 1) / SQUARE_BLOCKSIZE);
  MULTIPLY_SIGMA_INV<<<dimGrid, dimBlock>>>(m, n, dev_M, dev_V, dev_SIGMA_INV,
                                            dev_U);
}

__global__ void GET_RETENTION(int *k, int n, double *e, double *eigen_total,
                              double retention) {
  int k_retended = 0;
  double retention_done = 0;
  int i;

  for (i = 0; i < n; i++) {
    retention_done += 100 * e[i] / *eigen_total;
    k_retended++;
    if (retention_done >= retention) {
      break;
    }
  }

  *k = k_retended;
}

__global__ void GET_W(int k_retended, int n, double *W, double *E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n * k_retended) {
    W[i] = E[INDEX(i / k_retended, i % k_retended, n, n)];
  }
}

void SVD_and_PCA(int m, int n, double *D, double **U, double **SIGMA,
                 double **V_T, int *SIGMAm, int *SIGMAn, double **D_HAT, int *K,
                 int retention) {

  double *dev_M, *dev_M_T, *dev_S, *dev_e, *dev_E, *dev_new_E, *dev_eigen_total,
      *dev_SIGMA, *dev_SIGMA_INV, *dev_V_T, *dev_U, *dev_W, *dev_D_HAT;

  int *dev_k, *dev_indices,
      numblocks = (m * n + LINEAR_BLOCKSIZE - 1) / LINEAR_BLOCKSIZE;

  cudaMalloc(&dev_M, sizeof(double) * m * n);
  cudaMemcpy(dev_M, D, sizeof(double) * m * n, cudaMemcpyHostToDevice);
  cudaMalloc(&dev_M_T, sizeof(double) * m * n);

  TRANSPOSE<<<numblocks, LINEAR_BLOCKSIZE>>>(dev_M, m, n, dev_M_T);

  cudaMalloc(&dev_S, sizeof(double) * n * n);

  dim3 dimBlock(SQUARE_BLOCKSIZE, SQUARE_BLOCKSIZE);
  dim3 dimGrid((n + SQUARE_BLOCKSIZE - 1) / SQUARE_BLOCKSIZE,
               (n + SQUARE_BLOCKSIZE - 1) / SQUARE_BLOCKSIZE);

  MATMUL2<<<dimGrid, dimBlock>>>(n, m, n, dev_M_T, dev_M, dev_S);

  cudaFree(dev_M_T);

  cudaMalloc(&dev_e, sizeof(double) * n);
  cudaMalloc(&dev_E, sizeof(double) * n * n);
  JACOBI(n, dev_E, dev_e, dev_S);

  cudaFree(dev_S);

  cudaMalloc(&dev_indices, sizeof(int) * n);
  cudaMalloc(&dev_new_E, sizeof(double) * n * n);

  bool *converged;
  cudaMalloc(&converged, sizeof(bool));

  ODD_EVEN_SORT<<<1, LINEAR_BLOCKSIZE>>>(dev_e, dev_indices, n, converged);
  cudaFree(converged);

  numblocks = (n * n + LINEAR_BLOCKSIZE - 1) / LINEAR_BLOCKSIZE;
  ARRANGE<<<numblocks, LINEAR_BLOCKSIZE>>>(dev_indices, dev_E, dev_new_E, n, n);

  cudaFree(dev_indices);

  cudaFree(dev_E);
  dev_E = dev_new_E;

  cudaMalloc(&dev_SIGMA, sizeof(double) * n);
  cudaMalloc(&dev_SIGMA_INV, sizeof(double) * n);
  numblocks = (n + LINEAR_BLOCKSIZE - 1) / LINEAR_BLOCKSIZE;
  GET_SINGULAR_VALS<<<numblocks, LINEAR_BLOCKSIZE>>>(n, dev_e, dev_SIGMA,
                                                     dev_SIGMA_INV);

  cudaMalloc(&dev_eigen_total, sizeof(int));
  GET_EIGEN_SUM<<<1, 1>>>(dev_eigen_total, dev_e, n);

  cudaMalloc(&dev_V_T, sizeof(double) * n * n);
  numblocks = (n * n + LINEAR_BLOCKSIZE - 1) / LINEAR_BLOCKSIZE;
  TRANSPOSE<<<numblocks, LINEAR_BLOCKSIZE>>>(dev_E, n, n, dev_V_T);

  cudaMalloc(&dev_U, sizeof(double) * m * m);
  GET_U(m, n, dev_M, dev_E, dev_SIGMA_INV, dev_U);
  cudaFree(dev_SIGMA_INV);

  cudaMalloc(&dev_k, sizeof(int));
  GET_RETENTION<<<1, 1>>>(dev_k, n, dev_e, dev_eigen_total, retention);

  cudaFree(dev_eigen_total);
  cudaFree(dev_e);

  cudaMemcpy(K, dev_k, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(dev_k);

  cudaMalloc(&dev_W, sizeof(double) * n * (*K));
  cudaMalloc(&dev_D_HAT, sizeof(double) * m * (*K));

  numblocks = (n * (*K) + LINEAR_BLOCKSIZE - 1) / LINEAR_BLOCKSIZE;
  GET_W<<<numblocks, LINEAR_BLOCKSIZE>>>(*K, n, dev_W, dev_E);

  cudaFree(dev_E);

  dimGrid = dim3((*K + SQUARE_BLOCKSIZE - 1) / SQUARE_BLOCKSIZE,
                 (m + SQUARE_BLOCKSIZE - 1) / SQUARE_BLOCKSIZE);
  MATMUL2<<<dimGrid, dimBlock>>>(m, n, *K, dev_M, dev_W, dev_D_HAT);

  cudaFree(dev_W);
  cudaFree(dev_M);

  *U = (double *)malloc(sizeof(double) * m * m);
  cudaMemcpy(*U, dev_U, sizeof(double) * m * m, cudaMemcpyDeviceToHost);
  cudaFree(dev_U);

  *SIGMA = (double *)malloc(sizeof(double) * n);
  cudaMemcpy(*SIGMA, dev_SIGMA, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaFree(dev_SIGMA);

  *V_T = (double *)malloc(sizeof(double) * n * n);
  cudaMemcpy(*V_T, dev_V_T, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
  cudaFree(dev_V_T);

  *D_HAT = (double *)malloc(sizeof(double) * m * (*K));
  cudaMemcpy(*D_HAT, dev_D_HAT, sizeof(double) * m * (*K),
             cudaMemcpyDeviceToHost);

  // printMat<<<1, 1>>>(dev_D_HAT, m, *K);

  cudaFree(dev_D_HAT);

  cudaDeviceSynchronize();

  *SIGMAm = m;
  *SIGMAn = n;
}
