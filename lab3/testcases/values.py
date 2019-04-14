import sys
import numpy as np
from sklearn.decomposition import PCA

path = sys.argv[1]

with open(path) as f:
    vals = f.readline().split()
    m = int(vals[0])
    n = int(vals[1])
    mat = list(map(float, f.readline().split()))

mat = np.asarray(mat)

mat = mat.reshape((m, n))

print('M')
print(mat)

mat_T = mat.T

print('M Transpose')
print(mat_T)

S = np.dot(mat_T, mat)

print('S')
print(S)

w, v = np.linalg.eig(S)

print('Eigen Values')
print(w)

print(np.sum(w))

print('Eigen Vectors')
print(v)

smat_inv = np.zeros((n, m), dtype=float)
smat_inv[:n, :n] = np.diag(1/np.sqrt(w))

smat = np.zeros((m, n), dtype=float)
smat[:n, :n] = np.diag(np.sqrt(w))

u_reg = np.dot(np.dot(mat,v),smat_inv)
# np.set_printoptions(threshold=sys.maxsize)

# print('U by mult')
print(u_reg)
# print('M reg')
m_reg = np.dot(np.dot(u_reg,smat),v.T)
print('Max error in regeneration = ',np.max(np.absolute(np.array(mat)-np.array(m_reg))))

u, s, vh = np.linalg.svd(mat)
print('U')
# print(u)

print('SIGMA')
print(s)
print('SIGMA INV')
print(1/s)

print('VT')
print(vh)

pca = PCA(n_components=2)
y = pca.fit_transform(mat)
print(y)
