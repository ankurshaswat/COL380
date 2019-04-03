#include "lab3_cuda.h"

inline int INDEX(int i1, int i2, int l1, int l2) { return i1 * l2 + i2; }

int state, *ind, N;
bool *changed;
double *E, *e, *S;

int MAXIND(int k) {
  int m = k + 1, i;
  for (i = k + 2; i < N; i++) {
    if (fabs(S[INDEX(k, i, N, N)]) > fabs(S[INDEX(k, m, N, N)])) {
      m = i;
    }
  }
  return m;
}

__global__ void UPDATE(int k, double t) {
  double y = e[k];
  e[k] = y + t;
  if (changed[k] && y == e[k]) {
    changed[k] = false;
    state--;
  } else if (!changed[k] && !(y == e[k])) {
    changed[k] = true;
    state++;
  }
}

void ROTATE(int k, int l, int i, int j, double c, double s) {
  double Skl = S[INDEX(k, l, N, N)], Sij = S[INDEX(i, j, N, N)];
  S[INDEX(k, l, N, N)] = c * Skl - s * Sij;
  S[INDEX(i, j, N, N)] = s * Skl + c * Sij;
}

void JACOBI() {
  int i, j, k, m, l;
  double p, y, d, r, c, s, t, Eik, Eil;

  // Can be parallelized
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      E[INDEX(i, j, N, N)] = (i == j);
    }
  }

  // Single
  state = N;

  // Can be easily parallelized
  for (k = 0; k < N - 1; k++) {
    ind[k] = MAXIND(k);
    e[k] = S[INDEX(k, k, N, N)];
    changed[k] = true;
  }

  while (state != 0) {
    m = 0;

    // Parallelize
    for (k = 1; k < N; k++) {
      if (fabs(S[INDEX(k, ind[k], N, N)]) > fabs(S[INDEX(m, ind[m], N, N)])) {
        m = k;
      }
    }

    // Single
    k = m;
    l = ind[m];
    p = S[INDEX(k, l, N, N)];
    y = (e[l] - e[k]) / 2;
    d = fabs(y) + sqrt(p * p + y * y);
    r = sqrt(p * p + d * d);
    c = d / r;
    s = p / r;
    t = p * p / d;

    if (y < 0) {
      s = -s;
      t = -t;
    }

    S[INDEX(k, l, N, N)] = 0.0;

    UPDATE<<1,1>>(k, -t);
    UPDATE<<1,1>>(l, t);

    // int blockSize = 256;
    int numBlocks = (k-1 + BLOCKSIZE - 1) / BLOCKSIZE;

    // Can be parallelized
    ROTATE_MULTIPLE<<numBlocks,BLOCKSIZE>>()
    // for (i = 0; i < k; i++) {
      // ROTATE(i, k, i, l, c, s);
    // }

    // Can be parallelized
    for (i = k + 1; i < l; i++) {
      ROTATE(k, i, i, l, c, s);
    }
    
    // Can be parallelized
    for (i = l + 1; i < N; i++) {
      ROTATE(k, i, l, i, c, s);
    }

    // Parallelize Easily
    for (i = 0; i < N; i++) {
      Eik = E[INDEX(i, k, N, N)];
      Eil = E[INDEX(i, l, N, N)];
      E[INDEX(i, k, N, N)] = c * Eik - s * Eil;
      E[INDEX(i, l, N, N)] = s * Eik + c * Eil;
    }

    ind[k] = MAXIND(k);
    ind[l] = MAXIND(l);
  }
}

void TRANSPOSE(double *M, int m, int n, double *M_T) {
  int i, j;

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      M_T[INDEX(i, j, n, m)] = M[INDEX(j, i, m, n)];
    }
  }
}

void MATMUL(double *A, int p, int q, double *B, int r, double *C) {
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

void MATMUL_IN_ORDER(double *A, int p, int q, double *B_T, int r, double *C) {
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

void SORT(double *e, double *E, int n) {

  int *indices = (int *)malloc(sizeof(int) * n);
  int temp_int, i, j;

  double temp_double;
  double *old_E = E;
  double *new_E = (double *)malloc(sizeof(double) * n * n);

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

  free(indices);
  free(old_E);
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD_and_PCA(int m, int n, double *D, double **U, double **SIGMA,
                 double **V_T, double **D_HAT, int *K, int retention) {

  double *M = D, *M_T, *SIGMA_INV, *MV;
  double sum, sqrt_;
  double eigen_total = 0;
  int i, j, k;

  cudaMalloc(&M_T, sizeof(double) * m * n);
  cudaMalloc(&S, sizeof(double) * n * n);
  cudaMalloc(&E, sizeof(double) * n * n);
  cudaMalloc(&e, sizeof(double) * n);
  cudaMalloc(&SIGMA_INV, sizeof(double) * n);
  cudaMalloc(&MV, sizeof(double) * m * n);

  N = n;

  TRANSPOSE(M, m, n, M_T);

  MATMUL_IN_ORDER(M_T, n, m, M_T, n, S);

  JACOBI();

  SORT(e, E, n);

  /* Square root eigen values to get singular values and Compute SIGMA INVERSE*/
  for (i = 0; i < n; i++) {
    eigen_total += e[i];
    sqrt_ = sqrt(e[i]);
    (*SIGMA)[i] = sqrt_;
    SIGMA_INV[i] = 1 / sqrt_;
  }

  /* Compute V_T */
  TRANSPOSE(E, n, n, *V_T);

  /* Compute M.V.SIGMAINV */
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {

      sum = 0;
      if (j < n) {
        for (k = 0; k < n; k++) {
          sum += M[INDEX(i, k, m, n)] * (*V_T)[INDEX(j, k, n, n)];
        }
        sum *= SIGMA_INV[j];
      }

      (*U)[INDEX(i, j, m, m)] = sum;
    }
  }

  int k_retended = 0;
  double retention_done = 0;
  /* Choose k largest columns */
  for (i = 0; i < n; i++) {
    retention_done += 100 * e[i] / eigen_total;
    k_retended++;
    if (retention_done >= retention) {
      break;
    }
  }

  double *W = (double *)malloc(sizeof(double) * n * k_retended);
  /* Construct projection matrix */
  for (i = 0; i < k_retended; i++) {
    for (j = 0; j < n; j++) {
      W[INDEX(j, i, n, k_retended)] = E[INDEX(j, i, n, n)];
    }
  }

  /* Mat multiply to get d hat */
  MATMUL(D, m, n, W, k_retended, *D_HAT);

  *K = k;

  cudaFree(W);
  cudaFree(M_T);
  cudaFree(S);
  cudaFree(E);
  cudaFree(e);
  cudaFree(SIGMA_INV);
  cudaFree(V_T);
  cudaFree(MV);
}
