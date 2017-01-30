import numpy as np
import math
from sympy import Matrix
import GPy as gpy
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA
fac = math.factorial

# generate some sequential datas
pi = np.pi
x = np.linspace(0,2*pi,50)
# y = np.linspace(0,2*pi,10)
# x = x.reshape(20,1)
cosx = np.array(np.cos(x))
sinx = np.array(np.sin(x))
cosx = cosx.reshape(50,1)
sinx = sinx.reshape(50,1)
noise1 = np.random.normal(0, 1, 50)
noise1 = noise1.reshape(50,1)
noise2 = np.random.normal(0, 1, 20)
noise2 = noise2.reshape(20,1)
# sinnoisy = sinx + noise2
# xnoisy2 = x + noise2
y = np.linspace(0,100,50)
y = y.reshape(50,1)

cosines = []
labels = []
for i in range(0,10):
	noise = np.random.normal(0, 0.15, 50)
	noise = cosx + noise
	cosines.append(noise)
	labels.append(0)

for i in range(10,20):
	noise = np.random.normal(0, 0.15, 50)
	noise = sinx + noise
	cosines.append(noise)
	labels.append(1)

kernRBF = gpy.kern.RBF(1)
kernExp = gpy.kern.Exponential(1)

def k_gamma(i, j, Chv, Cd):
	sum = 0
	for d in range(0, min(i,j)-1):
		sum+= (Chv**(i+j-2-2*d))*(Cd**d)*((fac(i+j-2-d))/(fac(i-1-d)*fac(j-1-d)*fac(d)))
	return sum

def strucKernel(Cd, Chv, xlen, ylen):
	result = np.zeros((xlen, ylen))
	for i in range(1, xlen):
		for j in range(1, ylen):
			result[i-1][j-1] = k_gamma(i, j, Chv, Cd)
	return result

def pathKernel(s, t):
	# Ks = [len(s), len(t)]
	# slen = np.linspace(0,len(s),len(s))
	# tlen = np.linspace(0,len(t),len(t))
	# slen = slen.reshape(len(s),1)
	# tlen = tlen.reshape(len(t),1)
	kronMatrix = np.kron(kernRBF.K(s,t),strucKernel(0.4,0.3,len(s),len(t)))
	# plt.matshow(kronMatrix)
	# plt.show()
	total = kronMatrix.sum()
	return total

def kernelPCA(X, y, n_components):

	K = np.zeros((len(X),len(X)))
	for i, x1 in enumerate(X):
		for j, x2 in enumerate(X):
			K[i][j] = pathKernel(x1,x2)

	plt.matshow(K)
	# K = np.inner(cosines,cosines)
	# Centering the symmetric NxN kernel matrix.
	N = K.shape[0]
	one_n = np.ones((N,N)) / N
	K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

	# Obtaining eigenvalues in descending order with corresponding
	# eigenvectors from the symmetric matrix.
	eigvals, eigvecs = eigh(K)

	# Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
	X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
	# print(y==0)

	plt.figure(figsize=(8,6))
	plt.scatter(X_pc[:9, 0], X_pc[:9, 1], color='red', alpha=0.5)
	plt.scatter(X_pc[10:, 0], X_pc[10:, 1], color='blue', alpha=0.5)

	plt.title('First 2 principal components after Kernel PCA')
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.show()

# print(cosx)
# print(cosines)


# k_struc = strucKernel(0.35,0.37,40,40)
# plt.matshow(k_struc)
# plt.show()
# print(k_len)
# result = pathKernel(cosx,sinx)
# result2 = pathKernel(sinx,cosinx,kernRBF,kernExp)
# cosines = np.array(cosines)
# print(cosines.shape)

kernelPCA(cosines, labels, 2)
# print('kernel matrix: ', kmatrix)
# print('Cos and sin: ', result)
# print(result2)
