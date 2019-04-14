
// TODO
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

// TODO
__global__ void MATMUL(int p, int q, int r, double *A, double *B, double *C) {
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

__device__ double *GET_SUB_MATRIX_START_POS(double *mat, int w, int row,
                                            int col) {
  return &mat[w * BLOCKSIZE * row + BLOCKSIZE * col];
}

// TO TEST
__global__ void MATMUL_OPTIMIZED(int n1, int n2, int n3, double *A, double *B,
                                 double *C) {

  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;
  int row = threadIdx.x;
  int col = threadIdx.y;
  double Cvalue = 0;

  double *Csub = GET_SUB_MATRIX_START_POS(C, n3, blockRow, blockCol);
  // Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
  // printf("%d %d\n", n2, (n2 + BLOCKSIZE - 1) / BLOCKSIZE);

  for (int block_num = 0; block_num < ((n2 + BLOCKSIZE - 1) / BLOCKSIZE);
       block_num++) {
    double *Asub = GET_SUB_MATRIX_START_POS(A, n2, blockRow, block_num);
    // Matrix Asub = GetSubMatrix(A, blockRow, m);
    double *Bsub = GET_SUB_MATRIX_START_POS(B, n3, block_num, blockCol);
    // Matrix Bsub = GetSubMatrix(B, m, blockCol);

    __shared__ double As[BLOCKSIZE][BLOCKSIZE];
    __shared__ double Bs[BLOCKSIZE][BLOCKSIZE];

    As[row][col] = Asub[row * n2 + col];
    // As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = Bsub[row * n3 + col];
    // Bs[row][col] = GetElement(Bsub, row, col);

    __syncthreads();

    for (int e = 0; e < BLOCKSIZE; e++) {
      Cvalue += As[row][e] * Bs[e][col];
    }

    __syncthreads();
  }

  Csub[row * n3 + col] = Cvalue;
  // SetElement(Csub, row, col, Cvalue);
}

// TODO
__global__ void SORT(double *e, double *E, int n, int *indices, double *new_E) {

  int temp_int, i, j;

  double temp_double;
  double *old_E = E;

  for (i = 0; i < n; i++) {
    indices[i] = i;
  }

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (e[i] > e[j]) {
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

// // TODO
// __global__ void GET_U(int m, int n, double *M, double *V_T, double
// *SIGMA_INV,
//                       double *U) {
//   int i, j, k;
//   double sum;
//   /* Compute M.V.SIGMAINV */
//   for (i = 0; i < m; i++) {
//     for (j = 0; j < m; j++) {

//       sum = 0;
//       if (j < n) {
//         for (k = 0; k < n; k++) {
//           sum += M[INDEX(i, k, m, n)] * V_T[INDEX(j, k, n, n)];
//         }
//         sum *= SIGMA_INV[j];
//       }

//       U[INDEX(i, j, m, m)] = sum;
//     }
//   }
// }

// dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
// dim3 blocksPerGrid(n / threadsPerBlock.x, n / threadsPerBlock.y);
// MATMUL_OPTIMIZED<<<blocksPerGrid, threadsPerBlock>>>(n, m, n, dev_M_T,
// dev_M,
// dev_S);
//

// GET_U<<<1, 1>>>(m, n, dev_M, dev_V_T, dev_SIGMA_INV, dev_U);
// MATMUL_OPTIMIZED<<<numblocks,
// BLOCKSIZE>>>(m, n, *K, dev_M,
// dev_W, dev_D_HAT);