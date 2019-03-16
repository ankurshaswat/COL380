#include <iostream>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <vector>

using namespace std;

#define EPSILON 0.00001
#define CUTOFF 1000
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

  float D_T[n * m], *M_T = D, eigen_vectors[m * m], eigen_vectors_temp[m * m],
                    SIGMA_INV[n], M_V[n * m], A[m * m], Q_T[m * m], Q[m * m],
                    A_temp[m * m], R[m * m], u[m], diff = 0;

  float mod_u = 0;
  vector<pair<float, int>> eigen_values(m);

  int count = 0;
  float local_sum = 0, sum = 0;

#pragma omp parallel default(none)                                             \
    shared(m, n, eigen_vectors, R, D, D_T, M_T, A, count, sum, diff, u, Q,     \
           mod_u, eigen_vectors_temp, A_temp, eigen_values, SIGMA, SIGMA_INV,  \
           V_T, U, M_V, Q_T) firstprivate(local_sum) num_threads(NUM_THREADS)
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
        A[INDEX(i, j, m)] = sum;
      }
    }

    /* Get Eigen values and eigen vectors of M_T.M */
    while (count < CUTOFF) /* Break on convergence */
    {
#pragma omp barrier

#pragma omp single
      {
        count++;
        diff = 0;
      }

      /* Calculate Q and R */
      for (int i = 0; i < m; i++) {
        local_sum = 0;

#pragma omp for
        for (int j = 0; j < m; j++) {
          u[j] = A[INDEX(j, i, m)];
          for (int k = 0; k < i; k++) {
            u[j] -= R[INDEX(k, i, m)] * Q[INDEX(j, k, m)];
          }
          local_sum += u[j] * u[j];
        }

#pragma omp atomic
        sum += local_sum;

#pragma omp barrier

#pragma omp single
        {
          mod_u = sqrt(sum);
          sum = 0;
        }

#pragma omp for
        for (int k = 0; k < m; k++) {
          if (mod_u > EPSILON) {
            Q[INDEX(k, i, m)] = u[k] / mod_u;
            Q_T[INDEX(i, k, m)] = u[k] / mod_u;
          }
        }

#pragma omp for
        for (int j = i; j < m; j++) {

          R[INDEX(i, j, m)] = 0;
          for (int k = 0; k < m; k++) {
            R[INDEX(i, j, m)] += Q_T[INDEX(i, k, m)] * A[INDEX(k, j, m)];
          }
          
        }
      }

#pragma omp for collapse(2)
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {

          float sum1 = 0, sum2 = 0;
          for (int k = 0; k < n; k++) {
            sum1 += eigen_vectors[INDEX(i, k, m)] * Q_T[INDEX(j, k, m)];
            sum2 += R[INDEX(i, k, m)] * Q_T[INDEX(j, k, m)];
          }
          eigen_vectors_temp[INDEX(i, j, m)] = sum1;
          A_temp[INDEX(i, j, m)] = sum2;

        }
      }

#pragma omp for reduction(max : diff)
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          diff = fmax(fabs(eigen_vectors[INDEX(i, j, m)] -
                           eigen_vectors_temp[INDEX(i, j, m)]),
                      diff);
          diff = fmax(fabs(A[INDEX(i, j, m)] - A_temp[INDEX(i, j, m)]), diff);
          eigen_vectors[INDEX(i, j, m)] = eigen_vectors_temp[INDEX(i, j, m)];
          A[INDEX(i, j, m)] = A_temp[INDEX(i, j, m)];
        }
      }

      // #pragma omp master
      //       { printf("%.6f %d\n", diff, count); }

      /* Check for convergence and break */
      if (diff < EPSILON) {
        break;
      }
    }

/* Update eigen values */
#pragma omp for
    for (int i = 0; i < m; i++) {
      eigen_values[i].first = A[INDEX(i, i, m)];
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
        } else {
          (*U)[INDEX(i, j, n)] = 0;
        }
      }
    }
  }

  print(n, n, *U);
  print(m, *SIGMA);
  print(m, m, *V_T);
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int m, int n, float *D, float *U, float *SIGMA,
         float **D_HAT, int *K) {

  int k = 0;

  float stored_percentage = 0;

  for (int i = 0; i < n; i++) {
    stored_percentage += SIGMA[i];
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
}
