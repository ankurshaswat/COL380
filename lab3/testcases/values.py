import sys
import numpy as np

path = sys.argv[1]

with open(path) as f:
    vals = f.readline().split()
    m = int(vals[0])
    n= int(vals[1])
    mat = list(map(float,f.readline().split()))

mat = np.asarray(mat)

mat = mat.reshape((m,n))

print('M')
print(mat)

mat_T = mat.T

print('M Transpose')
print(mat_T)

S = np.dot(mat_T,mat)

print('S')
print(S)

w , v =np.linalg.eig(S)

print('Eigen Values')
print(w)

print('Eigen Vectors')
print(v)

u, s, vh = np.linalg.svd(mat)

print('U')
print(u)

print('VT')
print(vh)