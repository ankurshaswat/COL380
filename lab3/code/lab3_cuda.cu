#include "lab3_cuda.h"

#define BLOCKSIZE 32
#define MAXBLOCKS 65535
#define JACOBI_TOLERANCE 0.001

__device__ inline int INDEX(int i1, int i2, int l1, int l2) {
  return i1 * l2 + i2;
}

__global__ void printMat(double *mat, int n1, int n2) {
  printf("\n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", mat[INDEX(i, j, n1, n2)]);
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void printVec(double *vec, int n1) {
  printf("\n");
  for (int i = 0; i < n1; i++) {
    printf("%f ", vec[i]);
  }
  printf("\n");
  printf("\n");
}

__device__ void printVecDev(double *vec, int n1) {
  printf("\n");
  for (int i = 0; i < n1; i++) {
    printf("%f ", vec[i]);
  }
  printf("\n");
  printf("\n");
}

__global__ void printVec(bool *vec, int n1) {
  printf("\n");
  for (int i = 0; i < n1; i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
  printf("\n");
}

__global__ void printVec(int *vec, int n1) {
  printf("\n");
  for (int i = 0; i < n1; i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
  printf("\n");
}

// TODO
__device__ void MAXIND(int k, int N, double *S, int *result) {
  int m = k + 1, i;
  for (i = k + 2; i < N; i++) {
    if (fabsf(S[INDEX(k, i, N, N)]) > fabsf(S[INDEX(k, m, N, N)])) {
      m = i;
    }
  }
  *result = m;
}

__device__ void UPDATE(int k, double t, double *e, bool *changed, int *state) {
  double ek_prev = e[k];
  e[k] = ek_prev + t;

  if (e[k] < 0) {
    e[k] = 0;
  }

  if (changed[k] && (ek_prev - e[k]) < JACOBI_TOLERANCE) {
    changed[k] = false;
    (*state)--;
  } else if ((!changed[k]) && (ek_prev - e[k]) > JACOBI_TOLERANCE) {
    changed[k] = true;
    (*state)++;
  }
}

__device__ void ROTATE(int k, int l, int i, int j, double c, double s,
                       double *S, int N) {
  double Skl = S[INDEX(k, l, N, N)], Sij = S[INDEX(i, j, N, N)];
  S[INDEX(k, l, N, N)] = c * Skl - s * Sij;
  S[INDEX(i, j, N, N)] = s * Skl + c * Sij;
}

__global__ void INIT1(int N, double *E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N * N) {
    E[i] = ((i / N) == (i % N));
  }
}

__global__ void INIT2(int *state, int N) { *state = N; }

// TODO
__global__ void INIT3(int *ind, double *e, double *S, int N, bool *changed) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < N) {
    MAXIND(k, N, S, &ind[k]);
    e[k] = S[INDEX(k, k, N, N)];
    changed[k] = true;
  }
}

// TODO
__global__ void BEST_M(int *m, int N, double *S, int *ind) {
  *m = 0;
  int k;
  for (k = 1; k < N - 1; k++) {
    if (fabs(S[INDEX(k, ind[k], N, N)]) > fabs(S[INDEX(*m, ind[*m], N, N)])) {
      *m = k;
    }
  }
}

__global__ void GET_S_C(int *k, int *l, int *m, double *c, double *s, double *t,
                        int N, int *ind, double *S, double *e) {
  *k = *m;
  *l = ind[*m];
  double p = S[INDEX(*k, *l, N, N)];
  double y = (e[*l] - e[*k]) / 2;
  double d = fabs(y) + sqrt(p * p + y * y);
  double r = sqrt(p * p + d * d);

  *c = d / r;
  *s = p / r;
  *t = p * p / d;

  if (y < 0) {
    *s = -(*s);
    *t = -(*t);
  }

  S[INDEX(*k, *l, N, N)] = 0.0;
}

__global__ void UPDATE_COMBINED(int *k, int *l, double *t, double *e,
                                bool *changed, int *state) {
  UPDATE(*k, -1 * (*t), e, changed, state);
  UPDATE(*l, *t, e, changed, state);
}

__global__ void ROTATE_MULTIPLE1(int *k, int *l, double *c, double *s,
                                 double *S, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < *k) {
    ROTATE(i, *k, i, *l, *c, *s, S, N);
  }
}

__global__ void ROTATE_MULTIPLE2(int *k, int *l, double *c, double *s,
                                 double *S, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + (*k) + 1;

  if (i < *l) {
    ROTATE(*k, i, i, *l, *c, *s, S, N);
  }
}

__global__ void ROTATE_MULTIPLE3(int *k, int *l, double *c, double *s,
                                 double *S, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + (*l) + 1;

  if (i < N) {
    ROTATE(*k, i, *l, i, *c, *s, S, N);
  }
}

__global__ void UPDATE_E(int N, double *E, int *k, int *l, double *c,
                         double *s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double Eik, Eil;

  if (i < N) {
    Eik = E[INDEX(i, *k, N, N)];
    Eil = E[INDEX(i, *l, N, N)];
    E[INDEX(i, *k, N, N)] = (*c) * Eik - (*s) * Eil;
    E[INDEX(i, *l, N, N)] = (*s) * Eik + (*c) * Eil;
  }
}

__global__ void UPDATE_IND(int *k, int *l, int *ind, int N, double *S) {
  MAXIND(*k, N, S, &ind[*k]);
  MAXIND(*l, N, S, &ind[*l]);
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
    __syncthreads();
  }
}

__global__ void ARRANGE(int *indices, double *old_E, double *new_E, int n1,
                        int n2) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n1 * n2; i += stride) {
    new_E[i] = old_E[INDEX(i / n2, indices[i % n2], n1, n2)];
  }
}

__global__ void GET_SINGULAR_VALS(int n, double *e, double *SIGMA,
                                  double *SIGMA_INV) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i;
  double sqrt_;

  for (i = index; i < n; i += stride) {
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

void GET_U(int m, int n, double *dev_M, double *dev_V, double *dev_SIGMA_INV,
           double *dev_U) {
  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
  dim3 dimGrid((m + BLOCKSIZE - 1) / BLOCKSIZE,
               (m + BLOCKSIZE - 1) / BLOCKSIZE);
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

void JACOBI(int n, double *dev_E, double *dev_e, double *dev_S) {

  int *dev_state, *dev_ind, *dev_m, *dev_k, *dev_l;
  double *dev_c, *dev_s, *dev_t_;
  bool *dev_changed;
  int state = n;

  cudaMalloc(&dev_state, sizeof(int));
  cudaMalloc(&dev_ind, sizeof(int) * n);
  cudaMalloc(&dev_changed, sizeof(bool) * n);
  cudaMalloc(&dev_m, sizeof(int));
  cudaMalloc(&dev_k, sizeof(int));
  cudaMalloc(&dev_l, sizeof(int));
  cudaMalloc(&dev_c, sizeof(double));
  cudaMalloc(&dev_s, sizeof(double));
  cudaMalloc(&dev_t_, sizeof(double));

  int numblocks = (n * n + BLOCKSIZE - 1) / BLOCKSIZE;
  INIT1<<<numblocks, BLOCKSIZE>>>(n, dev_E);
  INIT2<<<1, 1>>>(dev_state, n);

  numblocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
  INIT3<<<numblocks, BLOCKSIZE>>>(dev_ind, dev_e, dev_S, n, dev_changed);

  int count = 0;

  while (state != 0 && count < 5 * n) {
    count++;

    // printf("%d %d\n", state, count);

    BEST_M<<<1, 1>>>(dev_m, n, dev_S, dev_ind);
    GET_S_C<<<1, 1>>>(dev_k, dev_l, dev_m, dev_c, dev_s, dev_t_, n, dev_ind,
                      dev_S, dev_e);
    UPDATE_COMBINED<<<1, 1>>>(dev_k, dev_l, dev_t_, dev_e, dev_changed,
                              dev_state);

    ROTATE_MULTIPLE1<<<numblocks, BLOCKSIZE>>>(dev_k, dev_l, dev_c, dev_s,
                                               dev_S, n);
    ROTATE_MULTIPLE2<<<numblocks, BLOCKSIZE>>>(dev_k, dev_l, dev_c, dev_s,
                                               dev_S, n);
    ROTATE_MULTIPLE3<<<numblocks, BLOCKSIZE>>>(dev_k, dev_l, dev_c, dev_s,
                                               dev_S, n);
    UPDATE_E<<<numblocks, BLOCKSIZE>>>(n, dev_E, dev_k, dev_l, dev_c, dev_s);
    UPDATE_IND<<<1, 1>>>(dev_k, dev_l, dev_ind, n, dev_S);

    cudaMemcpy(&state, dev_state, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }

  cudaFree(dev_state);
  cudaFree(dev_ind);
  cudaFree(dev_changed);
  cudaFree(dev_m);
  cudaFree(dev_k);
  cudaFree(dev_l);
  cudaFree(dev_c);
  cudaFree(dev_s);
  cudaFree(dev_t_);
}

void SVD_and_PCA(int m, int n, double *D, double **U, double **SIGMA,
                 double **V_T, int *SIGMAm, int *SIGMAn, double **D_HAT, int *K,
                 int retention) {

  double *dev_M, *dev_M_T, *dev_S, *dev_e, *dev_E, *dev_new_E, *dev_eigen_total,
      *dev_SIGMA, *dev_SIGMA_INV, *dev_V_T, *dev_U, *dev_W, *dev_D_HAT;

  int *dev_k, *dev_indices, numblocks = (m * n + BLOCKSIZE - 1) / BLOCKSIZE;

  cudaMalloc(&dev_M, sizeof(double) * m * n);
  cudaMemcpy(dev_M, D, sizeof(double) * m * n, cudaMemcpyHostToDevice);
  cudaMalloc(&dev_M_T, sizeof(double) * m * n);

  TRANSPOSE<<<numblocks, BLOCKSIZE>>>(dev_M, m, n, dev_M_T);

  cudaMalloc(&dev_S, sizeof(double) * n * n);

  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
  dim3 dimGrid((n + BLOCKSIZE - 1) / BLOCKSIZE,
               (n + BLOCKSIZE - 1) / BLOCKSIZE);

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
  numblocks = ((n / 2) + BLOCKSIZE - 1) / BLOCKSIZE;
  ODD_EVEN_SORT<<<numblocks, BLOCKSIZE>>>(dev_e, dev_indices, n, converged);
  cudaFree(converged);

  numblocks = (n * n + BLOCKSIZE - 1) / BLOCKSIZE;
  ARRANGE<<<numblocks, BLOCKSIZE>>>(dev_indices, dev_E, dev_new_E, n, n);
  cudaFree(dev_indices);

  cudaFree(dev_E);
  dev_E = dev_new_E;

  cudaMalloc(&dev_SIGMA, sizeof(double) * n);
  cudaMalloc(&dev_SIGMA_INV, sizeof(double) * n);
  numblocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
  GET_SINGULAR_VALS<<<numblocks, BLOCKSIZE>>>(n, dev_e, dev_SIGMA,
                                              dev_SIGMA_INV);

  cudaMalloc(&dev_eigen_total, sizeof(int));
  GET_EIGEN_SUM<<<1, 1>>>(dev_eigen_total, dev_e, n);

  cudaMalloc(&dev_V_T, sizeof(double) * n * n);
  numblocks = (n * n + BLOCKSIZE - 1) / BLOCKSIZE;
  TRANSPOSE<<<numblocks, BLOCKSIZE>>>(dev_E, n, n, dev_V_T);

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

  numblocks = (n * (*K) + BLOCKSIZE - 1) / BLOCKSIZE;
  GET_W<<<numblocks, BLOCKSIZE>>>(*K, n, dev_W, dev_E);

  cudaFree(dev_E);

  dimGrid =
      dim3((*K + BLOCKSIZE - 1) / BLOCKSIZE, (m + BLOCKSIZE - 1) / BLOCKSIZE);
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

  // printMat<<<1, 1>>>(dev_U, m, m);

  cudaFree(dev_D_HAT);

  cudaDeviceSynchronize();

  *SIGMAm = m;
  *SIGMAn = n;
}
