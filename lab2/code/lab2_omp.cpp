#include <iostream>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

using namespace std;

#define INDEX(i1, i2, l1) (i1 * l1 + i2)

void print(int m, int n, float *mat) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      cout << mat[INDEX(i, j, m)] << ' ';
    }
    cout << endl;
  }
}
void print(int m, float *row) {
  cout << endl;
  for (int i = 0; i < m; i++) {
    cout << row[i] << ' ';
  }
  cout << endl;
}
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int m, int n, float *D, float **U, float **SIGMA, float **V_T) {

  // D=M_T (MxN)
  // D_T = M (NxM)
  // M_T_M (MxM)
  // SIGMA,SIGMA_INV (MxM)
  // eigen values (Mx1)
  // num of eigen vectors = M
  //  eigen_vectors (MxM)
  // V (MxM)
  // M_V (NxM)

  float D_T[m][n];
  float *M_T = D;
  float M_T_M[m][m];
  float eigen_values[m];
  float eigen_vectors[m][m];
  float eigen_vectors_temp[m][m];
  float singular_vals[m];
  float SIGMA_INV[m];
  float V[n][m];
  float M_V[n][m];
  float A[m][m];

  print(m, n, D);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      eigen_vectors[i][j] = 0;
    }
  }
  //   print(m, m, eigen_vectors[0]);

  // Calculate D_T (an N x M matrix)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      D_T[i][j] = D[INDEX(j, i, m)];
    }
  }
  print(n, m, D_T[0]);

  // From here on consider D_T to be M (an N x M matrix). We already have M_T
  // which is D. Calculate M_T.M (an M x M matrix)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0;
      for (int k = 0; k < n; k++) {
        sum += M_T[INDEX(i, k, m)] * D_T[k][j];
      }
      M_T_M[i][j] = sum;
      A[i][j] = sum;
    }
  }
  print(m, m, M_T_M[0]);
  print(m, m, A[0]);

  // Get Eigen values and eigen vectors of M_T.M
  for (int i = 0; i < m; i++) {
    eigen_vectors[i][i] = 1;
  }

  float Q[m][m];
  float Q_temp[m][m];
  float R[m][m];
  float R_temp[m][m];

  while (true) // break on convergence
  {
    // calculate Q and R
    for (int i = 0; i < m; i++) {
      float u[m];
      float mod_u = 0;
      for (int j = 0; j < m; j++) {
        u[j] = A[i][j];
        for (int k = i - 1; k >= 0; k--) {
          u[j] -= R[k][i] * Q[j][k];
        }
        mod_u += u[j] * u[j];
      }
      mod_u = sqrt(mod_u);
      for (int j = 0; j < m; j++) {
        Q_temp[j][i] = u[j] / mod_u;
      }
      for (int j = 0; j < m; j++) {
        R_temp[i][j] = 0;
        for (int k = i; k < m; k++) {
          R_temp[i][j] += Q_temp[i][k] * A[j][k];
        }
      }
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        Q[i][j] = Q_temp[i][j];
        R[i][j] = R_temp[i][j];
      }
    }

    print(m,m, Q[0]);

    // Update eigen values and eigen vectors
    for (int i = 0; i < m; i++) {
      float sum = 0;
      for (int k = 0; k < n; k++) {
        sum += M_T[INDEX(i, k, m)] * D_T[k][i];
      }
      eigen_values[i] = sum;
    }

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
          sum += eigen_vectors[i][k] * Q[k][j];
        }
        eigen_vectors_temp[i][j] = sum;
      }
    }

    print(m, eigen_values);

    float diff = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        diff = fmax(fabs(eigen_vectors[i][j] - eigen_vectors_temp[i][j]), diff);
        eigen_vectors[i][j] = eigen_vectors_temp[i][j];
      }
    }

    // check for convergence and break
    if (diff < 0.0001) {
      break;
    }
  }

  // Sort Eigen values (and corresponding vectors) in descending order

  // Square root Eigen values to get Singular values. Put singular values along
  // diagonal in descending order to get SIGMA (an M x M diagonal matrix) Get
  // SIGMA_INV (an M x M diagonal matrix).
  for (int i = 0; i < m; i++) {
    float singular_val = sqrt(eigen_values[i]);
    (*SIGMA)[i] = singular_val;
    SIGMA_INV[i] = 1 / singular_val;
  }

  // Get V_T
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (i != j) {
        (*V_T)[INDEX(i, j, m)] = V[j][i];
      }
    }
  }

  // Get U = M.V.SIGMA_INV
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0;
      for (int k = 0; k < m; k++) {
        sum += D_T[i][k] * V[k][j];
      }
      M_V[i][j] = sum;
    }
  }

  // continue
}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float *D, float *U, float *SIGMA,
         float **D_HAT, int *K) {}
