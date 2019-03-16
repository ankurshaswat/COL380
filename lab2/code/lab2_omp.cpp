#include <iostream>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <vector>

using namespace std;

#define INDEX(i1, i2, l1) (i1 * l1 + i2)
#define EPSILON 0.00001

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

  float D_T[n][m];
  float *M_T = D;
  vector<pair<float, int>> eigen_values(m);
  float eigen_vectors[m][m];
  float eigen_vectors_temp[m][m];
  float SIGMA_INV[n];
  float V[m][m];
  float M_V[n][m];
  float A[m][m];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      eigen_vectors[i][j] = 0;
    }
  }

  // print(m,n,D);
  /* Calculate D_T (an N x M matrix) */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      D_T[i][j] = D[INDEX(j, i, n)];
    }
  }
  // print(n,m,D_T[0]);

  /* From here on consider D_T to be M (an N x M matrix). We already have M_T
   * which is D. Calculate M_T.M (an M x M matrix) */
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0;
      for (int k = 0; k < n; k++) {
        sum += M_T[INDEX(i, k, n)] * D_T[k][j];
      }
      A[i][j] = sum;
    }
  }
  // print(m, m, M_T_M[0]);
  print(m, m, A[0]);

  /* Get Eigen values and eigen vectors of M_T.M */
  for (int i = 0; i < m; i++) {
    eigen_vectors[i][i] = 1;
  }

  float Q[m][m], A_temp[m][m];
  float R[m][m] = {0};

  while (true) /* Break on convergence */
  {
    /* Calculate Q and R */
    for (int i = 0; i < m; i++) {
      float u[m];
      float mod_u = 0;
      for (int j = 0; j < m; j++) {
        u[j] = A[j][i];
        for (int k = i - 1; k >= 0; k--) {
          u[j] -= R[k][i] * Q[j][k];
        }
        mod_u += u[j] * u[j];
      }
      mod_u = sqrt(mod_u);
      if (mod_u > EPSILON) {
        for (int j = 0; j < m; j++) {
          Q[j][i] = u[j] / mod_u;
        }
      }

      for (int j = i; j < m; j++) {
        R[i][j] = 0;
        for (int k = 0; k < m; k++) {
          R[i][j] += Q[k][i] * A[k][j];
        }
      }
    }

    // cout<<"Q\n";
    // print(m, m, Q[0]);
    // cout<<"R\n";
    // print(m, m, R[0]);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float sum1 = 0, sum2 = 0;
        for (int k = 0; k < n; k++) {
          sum1 += eigen_vectors[i][k] * Q[k][j];
          sum2 += R[i][k] * Q[k][j];
        }
        eigen_vectors_temp[i][j] = sum1;
        A_temp[i][j] = sum2;
      }
    }

    float diff = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        diff = fmax(fabs(eigen_vectors[i][j] - eigen_vectors_temp[i][j]), diff);
        diff = fmax(fabs(A[i][j] - A_temp[i][j]), diff);
        eigen_vectors[i][j] = eigen_vectors_temp[i][j];
        A[i][j] = A_temp[i][j];
      }
    }

    // cout << diff << endl;

    /* Check for convergence and break */
    if (diff < EPSILON) {
      break;
    }
  }

  // /* Update eigen values */
  for (int i = 0; i < m; i++) {
    eigen_values[i].first = A[i][i];
    eigen_values[i].second = i;
  }

  /* Sort Eigen values (and corresponding vectors) in descending order */
  merge_sort(&eigen_values);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      V[i][j] = eigen_vectors[i][eigen_values[j].second];
    }
  }

  // print(m, &eigen_values);
  // print(m, m, eigen_vectors[0]);

  /* Square root Eigen values to get Singular values. Put singular values along
   * diagonal in descending order to get SIGMA (an M x M diagonal matrix) Get
   * SIGMA_INV (an M x M diagonal matrix). */

  for (int i = 0; i < n; i++) {
    float singular_val = sqrt(eigen_values[i].first);
    (*SIGMA)[i] = singular_val;
    SIGMA_INV[i] = 1 / singular_val;
  }
  print(n, *SIGMA);

  /* Get V_T */
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      (*V_T)[INDEX(i, j, m)] = V[j][i];
    }
  }

  /* Get U = M.V.SIGMA_INV */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0;
      for (int k = 0; k < m; k++) {
        sum += D_T[i][k] * V[k][j];
      }
      M_V[i][j] = sum;
      if (j < n) {
        (*U)[INDEX(i, j, n)] = M_V[i][j] * SIGMA_INV[j];
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
  *D_HAT = (float *)malloc(sizeof(float) * m * k);

  float W[n][k];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      W[i][j] = U[INDEX(i, j, n)];
    }
  }

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      float sum = 0;
      for (int y = 0; y < n; y++) {
        sum += D[INDEX(i, y, n)] * W[y][j];
      }
      (*D_HAT)[INDEX(i, j, k)] = sum;
    }
  }
}
