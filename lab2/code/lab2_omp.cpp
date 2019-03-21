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

void print(int m, int n, double *mat) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%.6f\t", mat[INDEX(i, j, n)]);
    }
    cout << endl;
  }
}

void print(int m, int n, float *mat) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%.6f\t", mat[INDEX(i, j, n)]);
    }
    cout << endl;
  }
}

void print(int m, vector<pair<double, int>> *row) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    cout << (*row)[i].first << ' ';
  }
  cout << endl;
}

void print(int m, double *row) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    cout << row[i] << ' ';
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

void printDiag(int m, double *mat, int n) {
  cout << endl;
  for (int i = 0; i < n; i++) {
    cout << mat[INDEX(i, i, m)] << ' ';
  }
  cout << endl;
}

void merge_sort(vector<pair<double, int>> *array) {
  int size = array->size();
  if (size <= 1) {
    return;
  } else if (size == 2) {
    if ((*array)[0] < (*array)[1]) {
      pair<double, int> temp = (*array)[0];
      (*array)[0] = (*array)[1];
      (*array)[1] = temp;
    }
    return;
  }

  int breaker = size / 2;

  vector<pair<double, int>> former, latter;

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
void SVD(int m, int n, float *D_in, float **U_in, float **SIGMA,
         float **V_T_in) {

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

  float *M = D_in;
  float **U2_T = V_T_in;
  float **V2 = U_in;
  double *V2_T = (double *)malloc(sizeof(double) * m * m);

  double *M_T = (double *)malloc(sizeof(double) * n * m);
  double *eigen_vectors = (double *)malloc(sizeof(double) * n * n);
  double *eigen_vectors_temp = (double *)malloc(sizeof(double) * n * n);
  double *A_T = (double *)malloc(sizeof(double) * n * n);
  double *Q_T = (double *)malloc(sizeof(double) * n * n);
  double *A_temp = (double *)malloc(sizeof(double) * n * n);
  double *R = (double *)malloc(sizeof(double) * n * n);
  double diff = 0, mod_u = 0, sum = 0;

  int count = 0;

  vector<pair<double, int>> eigen_values(n);

#pragma omp parallel default(none)                                             \
    shared(m, n, eigen_vectors, R, M_T, A_T, count, sum, diff, mod_u,          \
           eigen_vectors_temp, A_temp, eigen_values, SIGMA, Q_T, M, V2_T, V2,  \
           U2_T) num_threads(NUM_THREADS)
  {

#pragma omp for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        eigen_vectors[INDEX(i, j, n)] = (i == j);
        R[INDEX(i, j, n)] = 0;
      }
    }

#pragma omp for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) { /* Calculate D_T (an N x M matrix) */
        M_T[INDEX(i, j, m)] = M[INDEX(j, i, n)];
      }
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        /* From here on consider D_T to be M (an N x M matrix). We already
         * have M_T which is D. Calculate M_T.M (an M x M matrix) */
        double sum = 0;
        for (int k = 0; k < m; k++) {
          sum += M_T[INDEX(i, k, m)] * M_T[INDEX(j, k, m)];
        }
        A_T[INDEX(j, i, n)] = sum;
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
        /* Calculate Q and R */
        for (int i = 0; i < n; i++) {
          for (int k = 0; k < i; k++) {
            R[INDEX(k, i, n)] = 0;
            for (int j = 0; j < n; j++) {
              R[INDEX(k, i, n)] += Q_T[INDEX(k, j, n)] * A_T[INDEX(i, j, n)];
            }
            for (int j = 0; j < n; j++) {
              A_T[INDEX(i, j, n)] -= R[INDEX(k, i, n)] * Q_T[INDEX(k, j, n)];
            }
          }
          for (int j = 0; j < n; j++) {
            sum += A_T[INDEX(i, j, n)] * A_T[INDEX(i, j, n)];
          }
          mod_u = sqrt(sum);
          sum = 0;
          R[INDEX(i, i, n)] = mod_u;
          for (int j = 0; j < n; j++) {
            Q_T[INDEX(i, j, n)] = A_T[INDEX(i, j, n)] / mod_u;
          }
        }
      }
#pragma omp for collapse(2)
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

          double sum1 = 0, sum2 = 0;
          for (int k = 0; k < n; k++) {
            sum1 += eigen_vectors[INDEX(i, k, n)] * Q_T[INDEX(j, k, n)];
            sum2 += R[INDEX(i, k, n)] * Q_T[INDEX(j, k, n)];
          }
          eigen_vectors_temp[INDEX(i, j, n)] = sum1;
          A_temp[INDEX(i, j, n)] = sum2;
        }
      }

#pragma omp for collapse(2) reduction(max : diff)
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          diff = fmax(fabs(eigen_vectors[INDEX(i, j, n)] -
                           eigen_vectors_temp[INDEX(i, j, n)]),
                      diff);
          diff = fmax(fabs(A_T[INDEX(j, i, n)] - A_temp[INDEX(i, j, n)]), diff);
          // if (fabs(eigen_vectors_temp[INDEX(i, j, n)]) > EPSILON) {
          eigen_vectors[INDEX(i, j, n)] = eigen_vectors_temp[INDEX(i, j, n)];
          // } else {
          // eigen_vectors[INDEX(i, j, n)] = 0;
          // }

          // if (fabs(A_temp[INDEX(i, j, n)]) > EPSILON) {
          A_T[INDEX(j, i, n)] = A_temp[INDEX(i, j, n)];
          // } else {
          // A_T[INDEX(j, i, n)] = 0;
          // }
        }
      }

// #pragma omp master
//       {
//         printf("\n%.6f %d\n", diff, count);
//         // printDiag(n, A_T, n);
//       }

      /* Check for convergence and break */
      if (diff < EPSILON) {
        break;
      }
    }

/* Update eigen values */
#pragma omp for
    for (int i = 0; i < n; i++) {
      eigen_values[i].first = A_T[INDEX(i, i, n)];
      eigen_values[i].second = i;
    }

/* Sort Eigen values (and corresponding vectors) in descending order */
#pragma omp single
    merge_sort(&eigen_values);

#pragma omp for
    for (int i = 0; i < n; i++) {

      for (int j = 0; j < n; j++) {
        V2_T[INDEX(i, j, n)] =
            eigen_vectors[INDEX(j, eigen_values[i].second, n)];
        (*V2)[INDEX(j, i, n)] = V2_T[INDEX(i, j, n)];
      }

      // if (i < n) {
      /* Square root Eigen values to get Singular values. Put singular values
       * along diagonal in descending order to get SIGMA (an M x M diagonal
       * matrix) Get SIGMA_INV (an M x M diagonal matrix). */
      double singular_val = sqrt(eigen_values[i].first);
      (*SIGMA)[i] = singular_val;
      // SIGMA_INV[i] = 1 / singular_val;
      // }
    }

/* Get U = M.V.SIGMA_INV */
#pragma omp for collapse(2)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {

        double sum = 0;
        for (int k = 0; k < n; k++) {
          sum += M[INDEX(i, k, n)] * V2_T[INDEX(j, k, n)];
        }
        (*U2_T)[INDEX(j, i, m)] = sum * 1 / ((*SIGMA)[j]);
      }
    }
  }

  // double *M_CHECK = (double *)malloc(sizeof(double) * n * m);
  // double *U_SIG = (double *)malloc(sizeof(double) * n * m);

  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < m; j++) {

  //     double sum = 0;
  //     for (int k = 0; k < n; k++) {
  //       if (k == j) {
  //         sum += (*U_in)[INDEX(i, k, n)] * (*SIGMA)[k];
  //       }
  //     }
  //     U_SIG[INDEX(i, j, m)] = sum;
  //   }
  // }
  // double max_diff = 0;
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < m; j++) {

  //     double sum = 0;
  //     for (int k = 0; k < m; k++) {
  //       sum += U_SIG[INDEX(i, k, m)] * (*V_T_in)[INDEX(k, j, m)];
  //     }
  //     M_CHECK[INDEX(i, j, m)] = sum;
  //     max_diff = fmax(fabs(sum - D_in[INDEX(j, i, n)]), max_diff);
  //   }
  // }
  // printf("%f\n", max_diff);

  free(eigen_vectors);
  free(eigen_vectors_temp);
  free(A_T);
  free(Q_T);
  free(A_temp);
  free(R);
  free(V2_T);
  free(M_T);

  // print(n, n, *U_in);
  // print(n, *SIGMA);
  // print(m, m, *V_T_in);
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int m, int n, float *D, float *U, float *SIGMA,
         float **D_HAT, int *K) {

  int k = 0;
  double stored_percentage = 0, total_percentage = 0;

  for (int i = 0; i < n; i++) {
    total_percentage += SIGMA[i] * SIGMA[i];
  }

  for (int i = 0; i < n; i++) {
    stored_percentage += 100 * (SIGMA[i] * SIGMA[i]) / total_percentage;
    // cout << (SIGMA[i] * SIGMA[i]) / total_percentage << endl;
    k++;
    if (stored_percentage >= retention) {
      // printf("YO %f %f\n", stored_percentage,
      //        100 * (SIGMA[i + 1] * SIGMA[i + 1]) / total_percentage);
      break;
    }
  }

  *K = k;
  double W_T[k * n];
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
        double sum = 0;
        for (int y = 0; y < n; y++) {
          sum += D[INDEX(i, y, n)] * W_T[INDEX(j, y, n)];
        }
        (*D_HAT)[INDEX(i, j, k)] = sum;
      }
    }
  }

  // printf("K = %d\n", k);
  // print(n, SIGMA);
  // print(m,k,*D_HAT);
}
