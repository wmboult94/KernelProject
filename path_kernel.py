import numpy as np
import numpy.random as random
import math
import timeit
from sympy import Matrix
import GPy as gpy
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, kron
from sklearn.decomposition import KernelPCA
fac = math.factorial

###### generate some sequential datas
pi = np.pi
# x = np.linspace(0,2*pi,100)
# y = np.linspace(0,2*pi,10)
# x = x.reshape(20,1)
# cosx = np.array(np.cos(x))
# sinx = np.array(np.sin(x))
# cosx = cosx.reshape(100,1)
# sinx = sinx.reshape(100,1)
# noise1 = np.random.normal(0, 1, 20)
# noise1 = noise1.reshape(20,1)
# noise2 = np.random.normal(0, 1, 20)
# noise2 = noise2.reshape(20,1)
# sinnoisy = sinx + noise2
# xnoisy2 = x + noise2
# y = np.linspace(0,100,20)
# y = y.reshape(20,1)

cosines = []
labels = []
for i in range(0,10):
	size = random.random_integers(400,600)
	x = np.linspace(0,2*pi,size)
	x = x.reshape(size,1)
	cosx = np.array(np.cos(x))
	cosx = cosx.reshape(size,1)
	noise = random.normal(0, 0.25, size)
	noise = cosx + noise
	cosines.append(noise)
	labels.append(0)

for i in range(10,20):
	size = random.random_integers(400,600)
	x = np.linspace(0,2*pi,size)
	x = x.reshape(size,1)
	sinx = np.array(np.sin(x))
	sinx = sinx.reshape(size,1)
	noise = np.random.normal(0, 0.25, size)
	noise = sinx + noise
	cosines.append(noise)
	labels.append(1)

#######################################

kernRBF = gpy.kern.RBF(1)
kernExp = gpy.kern.Exponential(1)

def k_gamma(i, j, Chv, Cd):
	sum = 0
	for d in range(0, min(i,j)-1):
		sum+= (Chv**(i+j-2-2*d))*(Cd**d)*((fac(i+j-2-d))/(fac(i-1-d)*fac(j-1-d)*fac(d)))
	return sum

def strucKernel(Cd, Chv, xlen, ylen):
	# tvalue = np.inf if t==0 else t * min(xlen, ylen) + 1	# similar method to the triangular kga, only compute values near to diagonal
	result = np.zeros((xlen, ylen))
	for i in range(1, xlen):
		for j in range(1, ylen):
			# if (i-j < tvalue and i-j > -tvalue):
			result[i-1][j-1] = k_gamma(i, j, Chv, Cd)
			# else:
			# 	result[i-1][j-1] = 0
	return result

def pathKernel(s, t):
	# Ks = [len(s), len(t)]
	# slen = np.linspace(0,len(s),len(s))
	# tlen = np.linspace(0,len(t),len(t))
	# slen = slen.reshape(len(s),1)
	# tlen = tlen.reshape(len(t),1)
	kronMatrix = kron(kernRBF.K(s,t),strucKernel(0.4,0.3,len(s),len(t)))
	# plt.matshow(kronMatrix)
	# plt.show()
	total = kronMatrix.sum()
	return total

# Front kernel for sequences of same length, no overlap for now
#####
# Inputs: s, t are the input sequences; f is the front size; overlap is the number of diagonal steps overlap of the fronts
#####
def frontKernel(s, t, overlap):	# Front kernel can handle sequences of different lengths, don't use overlap for now when different lengths -- NB does overlap even really help in general?
	if len(s) != len(t): overlap=0
	f = 6
	# f = int(max(0.015*min(len(s), len(t)), 5))	# front size is the larger of 3/200 * length sequence, and 5
	acc=1
	j=1
	result=0
	inc_s = max(int(math.floor(float(len(s)) / max(len(s), len(t)) * f))-overlap, 1)
	inc_t = max(int(math.floor(float(len(t)) / max(len(s), len(t)) * f))-overlap, 1)
	for i in range(1, len(s)+1, inc_s):
		ub_s = min(i-1+f,len(s))
		ub_t = min(j-1+f,len(t))
		result += acc * pathKernel(s[i-1:ub_s], t[j-1:ub_t])
		# print(result)
		j += inc_t
		acc += 1
	return result

# Takes in an array of sequences X, builds the kernel matrix s.t K_ij = K(s_i,s_j)
def frontKernelMatrix(X, overlap):
	K = np.zeros((len(X),len(X)))
	for i, x1 in enumerate(X):
		for j, x2 in enumerate(X):
			# K[i][j] = pathKernel(x1,x2)
			K[i][j] = frontKernel(x1,x2,overlap)
	return K


def kernelPCA(X, y, n_components):

	K = frontKernelMatrix(X,0)

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

# functionality to time functions
def wrapper(func, *args, **kwargs):
	def wrapped():
		return func(*args, **kwargs)
	return wrapped

# wrapped1 = wrapper(pathKernel, cosx, sinx)
# wrapped2 = wrapper(frontKernel, cosx, sinx, 0)
# wrapped3 = wrapper(kernelPCA,cosines,labels,2)
# print('path kernel time: ', timeit.timeit(wrapped1))
# print('front kernel time: ', timeit.timeit(wrapped2))
# print('0.1 threshold val: ', timeit.timeit(wrapped3))

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
