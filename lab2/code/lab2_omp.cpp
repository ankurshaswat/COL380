#include <iostream>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <vector>

using namespace std;

#define EPSILON 0.0001
#define CUTOFF 100
#define NUM_THREADS 4

inline int INDEX(int i1, int i2, int l1) { return i1 * l1 + i2; }

void print(int m, int n, float *mat) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%.6f\t", mat[INDEX(i, j, n)]);
    }
    cout << endl;
  }
}

void print(int m, vector<pair<float, int>> *row) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    cout << (*row)[i].first << ' ';
  }
  cout << endl;
}

void print(int m, float *row) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    cout << row[i] << ' ';
  }
  cout << endl;
}

void printDiag(int m, float *mat, int n) {
  cout << endl;
  for (int i = 0; i < n; i++) {
    cout << mat[INDEX(i, i, m)] << ' ';
  }
  cout << endl;
}

void merge_sort(vector<pair<float, int>> *array) {
  int size = array->size();
  if (size <= 1) {
    return;
  } else if (size == 2) {
    if ((*array)[0] < (*array)[1]) {
      pair<float, int> temp = (*array)[0];
      (*array)[0] = (*array)[1];
      (*array)[1] = temp;
    }
    return;
  }

  int breaker = size / 2;

  vector<pair<float, int>> former, latter;

#pragma omp task default(none) shared(breaker, former, array)
  {
    for (int i = 0; i < breaker; i++) {
      former.push_back((*array)[i]);
    }
    merge_sort(&former);
  }

#pragma omp task default(none) shared(breaker, size, latter, array)
  {
    for (int i = breaker; i < size; i++) {
      latter.push_back((*array)[i]);
    }
    merge_sort(&latter);
  }

#pragma omp taskwait

  /* Merge Step */
  int p1 = 0, p2 = 0, p = 0;
  while (p1 < breaker && p2 < size - breaker) {
    if (former[p1].first > latter[p2].first) {
      (*array)[p++] = former[p1++];
    } else {
      (*array)[p++] = latter[p2++];
    }
  }
  while (p1 < breaker) {
    (*array)[p++] = former[p1++];
  }
  while (p2 < size - breaker) {
    (*array)[p++] = latter[p2++];
  }
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int m, int n, float *D, float **U, float **SIGMA, float **V_T) {

  /* D=M_T (MxN)
   * D_T = M (NxM)
   * M_T_M (MxM)
   * U (NxN)
   * SIGMA (NxM)
   * SIGMA_INV (MxN)
   * eigen values (Mx1)
   * num of eigen vectors = M
   * eigen_vectors (MxM)
   * V (MxM)
   * M_V (NxM) */

  float *M_T = D;
  float *D_T = (float *)malloc(sizeof(float) * n * m);
  float *eigen_vectors = (float *)malloc(sizeof(float) * m * m);
  float *eigen_vectors_temp = (float *)malloc(sizeof(float) * m * m);
  float *SIGMA_INV = (float *)malloc(sizeof(float) * n);
  float *M_V = (float *)malloc(sizeof(float) * n * m);
  float *A_T = (float *)malloc(sizeof(float) * m * m);
  float *Q_T = (float *)malloc(sizeof(float) * m * m);
  float *Q = (float *)malloc(sizeof(float) * m * m);
  float *A_temp = (float *)malloc(sizeof(float) * m * m);
  float *R = (float *)malloc(sizeof(float) * m * m);
  float *u = (float *)malloc(sizeof(float) * m);
  float diff = 0, mod_u = 0, sum = 0;

  int count = 0;

  vector<pair<float, int>> eigen_values(m);

#pragma omp parallel default(none)                                             \
    shared(m, n, eigen_vectors, R, D, D_T, M_T, A_T, count, sum, diff, u, Q,   \
           mod_u, eigen_vectors_temp, A_temp, eigen_values, SIGMA, SIGMA_INV,  \
           V_T, U, M_V, Q_T) num_threads(NUM_THREADS)
  {

#pragma omp for collapse(2)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {

        eigen_vectors[INDEX(i, j, m)] = (i == j);
        R[INDEX(i, j, m)] = 0;

        if (i < n) {
          /* Calculate D_T (an N x M matrix) */
          D_T[INDEX(i, j, m)] = D[INDEX(j, i, n)];
        }

        /* From here on consider D_T to be M (an N x M matrix). We already have
         * M_T which is D. Calculate M_T.M (an M x M matrix) */
        float sum = 0;
        for (int k = 0; k < n; k++) {
          sum += M_T[INDEX(i, k, n)] * D[INDEX(j, k, n)];
        }
        A_T[INDEX(j, i, m)] = sum;
      }
    }
    double c, s, mod, a1, a2, mem1, mem2;
    /* Get Eigen values and eigen vectors of M_T.M */
    while (count < CUTOFF) /* Break on convergence */
    {
#pragma omp barrier
#pragma omp single
      {
        count++;
        diff = 0;
        /* Calculate Q and R */
        // for (int j = 0; j < m; j++) {
        //   for (int i = j + 1; i < m; i++) {
        //     c = R[INDEX(j, j, m)];
        //     s = -1 * R[INDEX(i, j, m)];
        //     mod = sqrt(c * c + s * s);
        //     c /= mod;
        //     s /= mod;

        //     // do multiplication for q and r

        //     // #pragma omp parallel for
        //     for (int k = 0; k < m; k++) {
        //       a1 = R[INDEX(j, k, m)];
        //       a2 = R[INDEX(i, k, m)];
        //       R[INDEX(j, k, m)] = c * a1 - s * a2;
        //       R[INDEX(i, k, m)] = s * a1 + c * a2;

        //       mem1 = Q[INDEX(k, j, m)] * c - Q[INDEX(k, i, m)] * s;
        //       mem2 = Q[INDEX(k, i, m)] * c + Q[INDEX(k, j, m)] * s;
        //       Q[INDEX(k, j, m)] = mem1;
        //       Q[INDEX(k, i, m)] = mem2;
        //       Q_T[INDEX(j, k, m)] = mem1;
        //       Q_T[INDEX(i, k, m)] = mem2;
        //     }

        //     // print(m, m, Q);
        //     // print(m, m, R);
        //     R[INDEX(i, j, m)] = 0;
        //   }
        // }
        for (int i = 0; i < m; i++) {
          for (int k = 0; k < i; k++) {
            R[INDEX(k, i, m)] = 0;
            for (int j = 0; j < m; j++) {
              R[INDEX(k, i, m)] += Q_T[INDEX(k, j, m)] * A_T[INDEX(i, j, m)];
            }
            for (int j = 0; j < m; j++) {
              A_T[INDEX(i, j, m)] -= R[INDEX(k, i, m)] * Q_T[INDEX(k, j, m)];
            }
          }
          for (int j = 0; j < m; j++) {
            sum += A_T[INDEX(i, j, m)] * A_T[INDEX(i, j, m)];
          }
          mod_u = sqrt(sum);
          sum = 0;
          R[INDEX(i, i, m)] = mod_u;
          for (int j = 0; j < m; j++) {
            Q_T[INDEX(i, j, m)] = A_T[INDEX(i, j, m)] / mod_u;
          }
        }
      }
#pragma omp for collapse(2)
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {

          float sum1 = 0, sum2 = 0;
          for (int k = 0; k < m; k++) {
            sum1 += eigen_vectors[INDEX(i, k, m)] * Q_T[INDEX(j, k, m)];
            sum2 += R[INDEX(i, k, m)] * Q_T[INDEX(j, k, m)];
          }
          eigen_vectors_temp[INDEX(i, j, m)] = sum1;
          A_temp[INDEX(i, j, m)] = sum2;
        }
      }

#pragma omp for collapse(2) reduction(max : diff)
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          diff = fmax(fabs(eigen_vectors[INDEX(i, j, m)] -
                           eigen_vectors_temp[INDEX(i, j, m)]),
                      diff);
          diff = fmax(fabs(A_T[INDEX(j, i, m)] - A_temp[INDEX(i, j, m)]), diff);
          if (fabs(eigen_vectors_temp[INDEX(i, j, m)]) > EPSILON) {
            eigen_vectors[INDEX(i, j, m)] = eigen_vectors_temp[INDEX(i, j, m)];
          } else {
            eigen_vectors[INDEX(i, j, m)] = 0;
          }

          if (fabs(A_temp[INDEX(i, j, m)]) > EPSILON) {
            A_T[INDEX(j, i, m)] = A_temp[INDEX(i, j, m)];
          } else {
            A_T[INDEX(j, i, m)] = 0;
          }
        }
      }

#pragma omp master
      {
        printf("\n%.6f %d\n", diff, count);
        // printDiag(m, A_T, n + 3);
      }

      /* Check for convergence and break */
      if (diff < EPSILON) {
        break;
      }
    }

/* Update eigen values */
#pragma omp for
    for (int i = 0; i < m; i++) {
      eigen_values[i].first = A_T[INDEX(i, i, m)];
      eigen_values[i].second = i;
    }

/* Sort Eigen values (and corresponding vectors) in descending order */
#pragma omp single
    merge_sort(&eigen_values);

#pragma omp for
    for (int i = 0; i < m; i++) {

      for (int j = 0; j < m; j++) {
        (*V_T)[INDEX(i, j, m)] =
            eigen_vectors[INDEX(j, eigen_values[i].second, m)];
      }

      if (i < n) {
        /* Square root Eigen values to get Singular values. Put singular values
         * along diagonal in descending order to get SIGMA (an M x M diagonal
         * matrix) Get SIGMA_INV (an M x M diagonal matrix). */
        float singular_val = sqrt(eigen_values[i].first);
        (*SIGMA)[i] = singular_val;
        SIGMA_INV[i] = 1 / singular_val;
      }
    }

/* Get U = M.V.SIGMA_INV */
#pragma omp for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {

        float sum = 0;
        for (int k = 0; k < m; k++) {
          sum += D_T[INDEX(i, k, m)] * (*V_T)[INDEX(j, k, m)];
        }
        M_V[INDEX(i, j, m)] = sum;

        if (j < n) {
          (*U)[INDEX(i, j, n)] = M_V[INDEX(i, j, m)] * SIGMA_INV[j];
        }
      }
    }
  }

  free(D_T);
  free(eigen_vectors);
  free(eigen_vectors_temp);
  free(SIGMA_INV);
  free(M_V);
  free(A_T);
  free(Q_T);
  free(Q);
  free(A_temp);
  free(R);
  free(u);

  // print(n, n, *U);
  // print(n, *SIGMA);
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int m, int n, float *D, float *U, float *SIGMA,
         float **D_HAT, int *K) {

  int k = 0;
  float stored_percentage = 0, total_percentage = 0;

  for (int i = 0; i < n; i++) {
    total_percentage += SIGMA[i] * SIGMA[i];
  }

  for (int i = 0; i < n; i++) {
    stored_percentage += 100 * (SIGMA[i] * SIGMA[i]) / total_percentage;
    // cout << (SIGMA[i] * SIGMA[i]) / total_percentage << endl;
    k++;
    if (stored_percentage >= retention) {
      break;
    }
  }

  *K = k;
  float W_T[k * n];
  *D_HAT = (float *)malloc(sizeof(float) * m * k);

#pragma omp parallel default(none) shared(n, m, D_HAT, D, U, W_T, k)           \
    num_threads(NUM_THREADS)
  {

#pragma omp for collapse(2)
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        W_T[INDEX(i, j, n)] = U[INDEX(j, i, n)];
      }
    }

#pragma omp for collapse(2)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        float sum = 0;
        for (int y = 0; y < n; y++) {
          sum += D[INDEX(i, y, n)] * W_T[INDEX(j, y, n)];
        }
        (*D_HAT)[INDEX(i, j, k)] = sum;
      }
    }
  }

  printf("%d\n", k);
  print(n, SIGMA);
}
