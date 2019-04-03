#include "lab3_cuda.h"

__device__ inline int INDEX(int i1, int i2, int l1, int l2) {
  return i1 * l2 + i2;
}

__device__ void MAXIND(int k, int N, double *S, int *result) {
  int m = k + 1, i;
  for (i = k + 2; i < N; i++) {
    if (fabsf(S[INDEX(k, i, N, N)]) > fabsf(S[INDEX(k, m, N, N)])) {
      m = i;
    }
  }
  *result = m;
}

__global__ void UPDATE(int k, double t, double *e, bool *changed, int *state) {
  double y = e[k];
  e[k] = y + t;
  if (changed[k] && y == e[k]) {
    changed[k] = false;
    (*state)--;
  } else if (!changed[k] && !(y == e[k])) {
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
  int i, j;
  // Can be parallelized
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      E[INDEX(i, j, N, N)] = (i == j);
    }
  }
}

__global__ void INIT2(int *state, int N) {
  // Single
  *state = N;
}
__global__ void INIT3(int *ind, double *e, double *S, int N, bool *changed) {
  int k;
  // Can be easily parallelized
  for (k = 0; k < N - 1; k++) {
    MAXIND(k, N, S, &ind[k]);
    e[k] = S[INDEX(k, k, N, N)];
    changed[k] = true;
  }
}

__global__ void BEST_M(int *m, int N, double *S, int *ind) {
  *m = 0;
  int k;
  // Parallelize
  for (k = 1; k < N; k++) {
    if (fabs(S[INDEX(k, ind[k], N, N)]) > fabs(S[INDEX(*m, ind[*m], N, N)])) {
      *m = k;
    }
  }
}

__global__ void GET_S_C(int *k, int *l, int m, double *c, double *s, double *t,
                        int N, int *ind, double *S, double *e) {
  // Single
  *k = m;
  *l = ind[m];
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

__global__ void ROTATE_MULTIPLE1(int k, int l, double c, double s, double *S,
                                 int N) {
  int i;
  // Can be parallelized
  for (i = 0; i < k; i++) {
    ROTATE(i, k, i, l, c, s, S, N);
  }
}

__global__ void ROTATE_MULTIPLE2(int k, int l, double c, double s, double *S,
                                 int N) {
  int i;
  // Can be parallelized
  for (i = k + 1; i < l; i++) {
    ROTATE(k, i, i, l, c, s, S, N);
  }
}

__global__ void ROTATE_MULTIPLE3(int k, int l, double c, double s, double *S,
                                 int N) {
  int i;
  // Can be parallelized
  for (i = l + 1; i < N; i++) {
    ROTATE(k, i, l, i, c, s, S, N);
  }
}

__global__ void UPDATE_E(int N, double *E, int k, int l, double c, double s) {
  double Eik, Eil;
  int i;
  // Parallelize Easily
  for (i = 0; i < N; i++) {
    Eik = E[INDEX(i, k, N, N)];
    Eil = E[INDEX(i, l, N, N)];
    E[INDEX(i, k, N, N)] = c * Eik - s * Eil;
    E[INDEX(i, l, N, N)] = s * Eik + c * Eil;
  }
}

__global__ void UPDATE_IND(int k, int l, int *ind, int N, double *S) {
  MAXIND(k, N, S, &ind[k]);
  MAXIND(l, N, S, &ind[l]);
}

__global__ void TRANSPOSE(double *M, int m, int n, double *M_T) {
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      M_T[INDEX(i, j, n, m)] = M[INDEX(j, i, m, n)];
    }
  }
}

__global__ void MATMUL(double *A, int p, int q, double *B, int r, double *C) {
  int i, j, k;

  for (i = 0; i < p; i++) {
    for (j = 0; j < r; j++) {
      C[INDEX(i, j, p, r)] = 0;
      for (k = 0; k < q; k++) {
        C[INDEX(i, j, p, r)] += A[INDEX(i, k, p, q)] * B[INDEX(k, j, q, r)];
      }
    }
  }
}

__global__ void MATMUL_IN_ORDER(double *A, int p, int q, double *B_T, int r,
                                double *C) {
  int i, j, k;

  for (i = 0; i < p; i++) {
    for (j = 0; j < r; j++) {
      C[INDEX(i, j, p, r)] = 0;
      for (k = 0; k < q; k++) {
        C[INDEX(i, j, p, r)] += A[INDEX(i, k, p, q)] * B_T[INDEX(j, k, r, q)];
      }
    }
  }
}

__global__ void SORT(double *e, double *E, int n, int *indices, double *new_E) {

  int temp_int, i, j;

  double temp_double;
  double *old_E = E;

  for (i = 0; i < n; i++) {
    indices[i] = i;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (e[i] < e[j]) {
        temp_double = e[i];
        e[i] = e[j];
        e[j] = temp_double;

        temp_int = indices[i];
        indices[i] = indices[j];
        indices[j] = temp_int;
      }
    }
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      new_E[INDEX(j, i, n, n)] = old_E[INDEX(j, indices[i], n, n)];
    }
  }
}

void JACOBI(int n, double *dev_E, double *dev_e, double *dev_S) {

  int *dev_state, *dev_ind, *dev_m, *dev_k, *dev_l;
  double *dev_c, *dev_s, *dev_t_;
  bool *dev_changed;

  cudaMalloc(&dev_E, sizeof(double) * n * n);
  cudaMalloc(&dev_state, sizeof(int));
  cudaMalloc(&dev_ind, sizeof(int) * n);
  cudaMalloc(&dev_e, sizeof(double) * n);
  cudaMalloc(&dev_changed, sizeof(bool) * n);
  cudaMalloc(&dev_m, sizeof(int));
  cudaMalloc(&dev_k, sizeof(int));
  cudaMalloc(&dev_l, sizeof(int));
  cudaMalloc(&dev_c, sizeof(double));
  cudaMalloc(&dev_s, sizeof(double));
  cudaMalloc(&dev_t_, sizeof(double));

  INIT1<<<1, 1>>>(n, dev_E);
  INIT2<<<1, 1>>>(dev_state, n);
  INIT3<<<1, 1>>>(dev_ind, dev_e, dev_S, n, dev_changed);

  while (dev_state != 0) {

    BEST_M<<<1, 1>>>(dev_m, n, dev_S, dev_ind);

    GET_S_C<<<1, 1>>>(dev_k, dev_l, *dev_m, dev_c, dev_s, dev_t_, n, dev_ind,
                      dev_S, dev_e);

    UPDATE<<<1, 1>>>(*dev_k, -(*dev_t_), dev_e, dev_changed, dev_state);
    UPDATE<<<1, 1>>>(*dev_l, *dev_t_, dev_e, dev_changed, dev_state);

    ROTATE_MULTIPLE1<<<1, 1>>>(*dev_k, *dev_l, *dev_c, *dev_s, dev_S, n);
    ROTATE_MULTIPLE2<<<1, 1>>>(*dev_k, *dev_l, *dev_c, *dev_s, dev_S, n);
    ROTATE_MULTIPLE3<<<1, 1>>>(*dev_k, *dev_l, *dev_c, *dev_s, dev_S, n);

    UPDATE_E<<<1, 1>>>(n, dev_E, *dev_k, *dev_l, *dev_c, *dev_s);
    UPDATE_IND<<<1, 1>>>(*dev_k, *dev_l, dev_ind, n, dev_S);
  }

  cudaFree(&dev_E);
  cudaFree(&dev_state);
  cudaFree(&dev_ind);
  cudaFree(&dev_e);
  cudaFree(&dev_changed);
  cudaFree(&dev_m);
  cudaFree(&dev_k);
  cudaFree(&dev_l);
  cudaFree(&dev_c);
  cudaFree(&dev_s);
  cudaFree(&dev_t_);
}

__global__ void GET_SINGULAR_VALS(int n, double *eigen_total, double *e,
                                  double *SIGMA, double *SIGMA_INV) {
  int i;
  double sqrt_;
  *eigen_total = 0;
  /* Square root eigen values to get singular values and Compute SIGMA INVERSE*/
  for (i = 0; i < n; i++) {
    *eigen_total += e[i];
    sqrt_ = sqrt(e[i]);
    SIGMA[i] = sqrt_;
    SIGMA_INV[i] = 1 / sqrt_;
  }
}

__global__ void GET_U(int m, int n, double *M, double *V_T, double *SIGMA_INV,
                      double *U) {
  int i, j, k;
  double sum;
  /* Compute M.V.SIGMAINV */
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {

      sum = 0;
      if (j < n) {
        for (k = 0; k < n; k++) {
          sum += M[INDEX(i, k, m, n)] * V_T[INDEX(j, k, n, n)];
        }
        sum *= SIGMA_INV[j];
      }

      U[INDEX(i, j, m, m)] = sum;
    }
  }
}

__global__ void GET_W(int k_retended, int n, double *W, double *E) {
  int i, j;
  /* Construct projection matrix */
  for (i = 0; i < k_retended; i++) {
    for (j = 0; j < n; j++) {
      W[INDEX(j, i, n, k_retended)] = E[INDEX(j, i, n, n)];
    }
  }
}

__global__ void GET_RETENTION(int *k, int n, double *e, double eigen_total,
                              double retention) {
  int k_retended = 0;
  double retention_done = 0;
  /* Choose k largest columns */
  int i;
  for (i = 0; i < n; i++) {
    retention_done += 100 * e[i] / eigen_total;
    k_retended++;
    if (retention_done >= retention) {
      break;
    }
  }

  *k = k_retended;
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD_and_PCA(int m, int n, double *D, double **U, double **SIGMA,
                 double **V_T, double **D_HAT, int *K, int retention) {

  double *dev_M, *dev_M_T, *dev_S, *dev_e, *dev_E, *dev_new_E, *dev_eigen_total,
      *dev_SIGMA, *dev_SIGMA_INV, *dev_V_T, *dev_U, *dev_W, *dev_D_HAT;

  int *dev_k, *dev_indices;

  cudaMalloc(&dev_M, sizeof(double) * m * n);
  cudaMalloc(&dev_M_T, sizeof(double) * m * n);
  cudaMalloc(&dev_S, sizeof(double) * n * n);

  cudaMemcpy(dev_M, D, sizeof(double) * m * n, cudaMemcpyHostToDevice);

  TRANSPOSE<<<1, 1>>>(dev_M, m, n, dev_M_T);

  MATMUL_IN_ORDER<<<1, 1>>>(dev_M_T, n, m, dev_M_T, n, dev_S);

  cudaFree(dev_M_T);

  cudaMalloc(&dev_e, sizeof(double) * n);
  cudaMalloc(&dev_E, sizeof(double) * n * n);

  JACOBI(n, dev_E, dev_e, dev_S);

  cudaFree(dev_S);

  cudaMalloc(&dev_indices, sizeof(int) * n);
  cudaMalloc(&dev_new_E, sizeof(double) * n * n);

  SORT<<<1, 1>>>(dev_e, dev_E, n, dev_indices, dev_new_E);

  cudaFree(dev_indices);
  cudaFree(dev_E);
  dev_E = dev_new_E;

  cudaMalloc(&dev_eigen_total, sizeof(int));
  cudaMalloc(&dev_SIGMA, sizeof(double) * n);
  cudaMalloc(&dev_SIGMA_INV, sizeof(double) * n);
  cudaMalloc(&dev_V_T, sizeof(double) * n * n);
  cudaMalloc(&dev_U, sizeof(double) * m * m);

  GET_SINGULAR_VALS<<<1, 1>>>(n, dev_eigen_total, dev_e, dev_SIGMA,
                              dev_SIGMA_INV);

  /* Compute V_T */
  TRANSPOSE<<<1, 1>>>(dev_E, n, n, dev_V_T);

  GET_U<<<1, 1>>>(m, n, dev_M, dev_V_T, dev_SIGMA_INV, dev_U);

  cudaFree(dev_SIGMA_INV);

  cudaMalloc(&dev_k, sizeof(int));

  // double *dev_retention;

  GET_RETENTION<<<1, 1>>>(dev_k, n, dev_e, *dev_eigen_total, retention);

  cudaMemcpy(K, dev_k, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_e);
  cudaFree(dev_eigen_total);
  cudaFree(dev_k);

  cudaMalloc(&dev_W, sizeof(double) * n * (*K));

  GET_W<<<1, 1>>>(*K, n, dev_W, dev_E);

  cudaFree(dev_E);
  cudaMalloc(&dev_D_HAT, sizeof(double) * m * (*K));

  /* Mat multiply to get d hat */
  MATMUL<<<1, 1>>>(dev_M, m, n, dev_W, *dev_k, dev_D_HAT);

  *D_HAT = (double *)malloc(sizeof(double) * m * (*K));

  cudaMemcpy(*U, dev_U, sizeof(double) * m * m, cudaMemcpyDeviceToHost);
  cudaMemcpy(*SIGMA, dev_SIGMA, sizeof(double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(*V_T, dev_V_T, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(*D_HAT, dev_D_HAT, sizeof(double) * m * (*K),
             cudaMemcpyDeviceToHost);

  cudaFree(dev_M);
  cudaFree(dev_W);
  cudaFree(dev_U);
  cudaFree(dev_SIGMA);
  cudaFree(dev_V_T);
  cudaFree(dev_D_HAT);
  // *K = k;
}
