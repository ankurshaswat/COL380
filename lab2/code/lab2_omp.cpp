#include <iostream>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <vector>

using namespace std;

#define EPSILON 0.00001
#define CUTOFF 1000

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
  }

  int breaker = size / 2;

  vector<pair<float, int>> former, latter;

  for (int i = 0; i < breaker; i++) {
    former.push_back((*array)[i]);
  }
  merge_sort(&former);

  for (int i = breaker; i < size; i++) {
    latter.push_back((*array)[i]);
  }
  merge_sort(&latter);

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
                    SIGMA_INV[n], V[m * m], M_V[n * m], A[m * m], Q[m * m],
                    A_temp[m * m], R[m * m];

  vector<pair<float, int>> eigen_values(m);

  // #pragma omp parallel default(none)                                             \
    // shared(m, n, eigen_vectors, R, D, D_T, M_T, A) num_threads(1)
  {
    // #pragma omp for collapse(2)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        eigen_vectors[INDEX(i, j, m)] = 0;
        R[INDEX(i, j, m)] = 0;
      }
    }

    /* Calculate D_T (an N x M matrix) */
    // #pragma omp for collapse(2)
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        D_T[INDEX(i, j, m)] = D[INDEX(j, i, n)];
      }
    }

    /* From here on consider D_T to be M (an N x M matrix). We already have M_T
     * which is D. Calculate M_T.M (an M x M matrix) */
    // #pragma omp for collapse(2)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
          sum += M_T[INDEX(i, k, n)] * D[INDEX(j, k, n)];
        }
        A[INDEX(i, j, m)] = sum;
      }
    }

    /* Get Eigen values and eigen vectors of M_T.M */
    // #pragma omp for
    for (int i = 0; i < m; i++) {
      eigen_vectors[INDEX(i, i, m)] = 1;
    }
  }

  int count = 0;
  while (count < CUTOFF) /* Break on convergence */
  {
    count++;
    /* Calculate Q and R */
    for (int i = 0; i < m; i++) {
      float u[m];
      float mod_u = 0;
      for (int j = 0; j < m; j++) {
        u[j] = A[INDEX(j, i, m)];
        for (int k = i - 1; k >= 0; k--) {
          u[j] -= R[INDEX(k, i, m)] * Q[INDEX(j, k, m)];
        }
        mod_u += u[j] * u[j];
      }
      mod_u = sqrt(mod_u);
      if (mod_u > EPSILON) {
        for (int j = 0; j < m; j++) {
          Q[INDEX(j, i, m)] = u[j] / mod_u;
        }
      }

      for (int j = i; j < m; j++) {
        R[INDEX(i, j, m)] = 0;
        for (int k = 0; k < m; k++) {
          R[INDEX(i, j, m)] += Q[INDEX(k, i, m)] * A[INDEX(k, j, m)];
        }
      }
    }

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float sum1 = 0, sum2 = 0;
        for (int k = 0; k < n; k++) {
          sum1 += eigen_vectors[INDEX(i, k, m)] * Q[INDEX(k, j, m)];
          sum2 += R[INDEX(i, k, m)] * Q[INDEX(k, j, m)];
        }
        eigen_vectors_temp[INDEX(i, j, m)] = sum1;
        A_temp[INDEX(i, j, m)] = sum2;
      }
    }

    float diff = 0;
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
    // cout<<diff<<' '<<count<<endl;

    /* Check for convergence and break */
    if (diff < EPSILON) {
      break;
    }
  }

  // /* Update eigen values */
  for (int i = 0; i < m; i++) {
    eigen_values[i].first = A[INDEX(i, i, m)];
    eigen_values[i].second = i;
  }

  /* Sort Eigen values (and corresponding vectors) in descending order */
  merge_sort(&eigen_values);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      V[INDEX(i, j, m)] = eigen_vectors[INDEX(i, eigen_values[j].second, m)];
    }
  }

  /* Square root Eigen values to get Singular values. Put singular values
   * along diagonal in descending order to get SIGMA (an M x M diagonal
   * matrix) Get SIGMA_INV (an M x M diagonal matrix). */

  for (int i = 0; i < n; i++) {
    float singular_val = sqrt(eigen_values[i].first);
    (*SIGMA)[i] = singular_val;
    SIGMA_INV[i] = 1 / singular_val;
  }

  /* Get V_T */
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      (*V_T)[INDEX(i, j, m)] = V[INDEX(j, i, m)];
    }
  }

  /* Get U = M.V.SIGMA_INV */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0;
      for (int k = 0; k < m; k++) {
        sum += D_T[INDEX(i, k, m)] * V[INDEX(k, j, m)];
      }
      M_V[INDEX(i, j, m)] = sum;
      if (j < n) {
        (*U)[INDEX(i, j, n)] = M_V[INDEX(i, j, m)] * SIGMA_INV[j];
      } else {
        (*U)[INDEX(i, j, n)] = 0;
      }
    }
  }
  // print(n, n, *U);
  // print(m, *SIGMA);
  // print(m, m, *V_T);
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
  float W[n * k];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      W[INDEX(i, j, k)] = U[INDEX(i, j, n)];
    }
  }

  *D_HAT = (float *)malloc(sizeof(float) * m * k);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      float sum = 0;
      for (int y = 0; y < n; y++) {
        sum += D[INDEX(i, y, n)] * W[INDEX(y, j, k)];
      }
      (*D_HAT)[INDEX(i, j, k)] = sum;
    }
  }
}
