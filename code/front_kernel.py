import sys
import os
import io
import cPickle as pickle

import numpy as np
import collections
import numpy.random as random
import math
import time
from sympy import Matrix
import GPy as gpy
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, kron
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score,train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn

pi = np.pi
fac = math.factorial
kernRBF = gpy.kern.RBF(1)
# linearKern = gpy.kern.linear(1)
kernExp = gpy.kern.Exponential(1)

#-----------------#

def kGamma(i, j, Chv, Cd):
	sum = 0
	for d in range(0, min(i,j)-1):
		sum+= (Chv**(i+j-2-2*d))*(Cd**d)*((fac(i+j-2-d))/(fac(i-1-d)*fac(j-1-d)*fac(d)))
	return sum

def strucKernel(Cd, Chv, xlen, ylen, n):
	# tvalue = np.inf if t==0 else t * min(xlen, ylen) + 1	# similar method to the triangular kga, only compute values near to diagonal
	result = np.zeros((xlen, ylen))
	for i in range(1, xlen):
		for j in range(1, ylen):
			# if (i-j < tvalue and i-j > -tvalue):
			result[i-1][j-1] = kGamma(i, j, Chv, Cd) * pow(kGamma(xlen, ylen, 1, 1), n)
			# else:
			# 	result[i-1][j-1] = 0
	return result

def pathKernel(s, t, n=1):
	# Ks = [len(s), len(t)]
	kronMatrix = kron(kernRBF.K(s,t),strucKernel(0.35,0.33,len(s),len(t), n))
	# kronMatrix = kron(kernRBF.K(s,t),kernExp.K(np.arange(len(s)).reshape(len(s),1),np.arange(len(t)).reshape(len(t),1)))
	# plt.matshow(kronMatrix)
	# plt.show()
	total = kronMatrix.sum()
	return total

#*** Front kernel ***#
#####
# Inputs: s, t are the input sequences; f is the front size; overlap is the number of diagonal steps overlap of the fronts
#####
def frontKernel(s, t, f=-1, overlap=0):	# Front kernel can handle sequences of different lengths

	if len(s) != len(t): overlap=0
	if f == -1:
		f_s = int(min(15,max(0.05*len(s), 5)))	# front size s length is between 5 and 15
		f_t = int(min(15,max(0.05*len(t), 5)))	# front size t length is between 5 and 15
	else:
		f_s = f
		f_t = f

	acc=0
	j=1
	result=0
	max_len = max(len(s), len(t))
	inc_s = max(int(math.floor(float(len(s)) / max_len * f_s))-overlap, 1)	# work out front increments for each sequence
	inc_t = max(int(math.floor(float(len(t)) / max_len * f_t))-overlap, 1)
	inc_i = max(inc_s, inc_t)
	inc_j = min(inc_s, inc_t)
	for i in range(1, max_len+1, inc_i):
		ub_s = min(i-1+f_s,len(s))
		ub_t = min(j-1+f_t,len(t))
		result += pathKernel(s[i-1:ub_s], t[j-1:ub_t], acc)
		# print(result)
		j += inc_j
		acc += 1
	return result

# Build a front kernel for 2 dimensional co-ordinate data :- K(i,j) is a sum of the individual co-ordinates, K(i[xs,ys],j[xs,ys]) = K(i[xs],j[xs]) + K(i[ys],j[ys])
def twoDimFrontMatrix(X,Y=None):
	if Y == None:
		K = np.zeros((len(X),len(X)))
		for i, c1 in enumerate(X):
			print '***********'
			print i, '/', len(X)
			for j, c2 in enumerate(X):
				x1 = c1[:,0].reshape(len(c1[:,0]),1)
				x2 = c2[:,0].reshape(len(c2[:,0]),1)
				y1 = c1[:,1].reshape(len(c1[:,1]),1)
				y2 = c2[:,1].reshape(len(c2[:,1]),1)

				# K[i][j] = pathKernel(x1,x2)
				K[i][j] = frontKernel(x1,x2) + frontKernel(y1,y2)
			print '\n'
	else:
		K = np.zeros((len(X),len(Y)))
		for i, c1 in enumerate(X):
			print '***********'
			print i, '/', len(X)
			for j, c2 in enumerate(Y):
				x1 = c1[:,0].reshape(len(c1[:,0]),1)
				x2 = c2[:,0].reshape(len(c2[:,0]),1)
				y1 = c1[:,1].reshape(len(c1[:,1]),1)
				y2 = c2[:,1].reshape(len(c2[:,1]),1)

				# K[i][j] = pathKernel(x1,x2)
				K[i][j] = frontKernel(x1,x2) + frontKernel(y1,y2)
			print '\n'
	return K

# For timing comparisons
def twoDimPathMatrix(X,Y=None):
	if Y == None:
		K = np.zeros((len(X),len(X)))
		for i, c1 in enumerate(X):
			print '***********'
			print i, '/', len(X)
			for j, c2 in enumerate(X):
				x1 = c1[:,0].reshape(len(c1[:,0]),1)
				x2 = c2[:,0].reshape(len(c2[:,0]),1)
				y1 = c1[:,1].reshape(len(c1[:,1]),1)
				y2 = c2[:,1].reshape(len(c2[:,1]),1)

				# K[i][j] = pathKernel(x1,x2)
				K[i][j] = pathKernel(x1,x2) + pathKernel(y1,y2)
			print '\n'
	else:
		K = np.zeros((len(X),len(Y)))
		for i, c1 in enumerate(X):
			print '***********'
			print i, '/', len(X)
			for j, c2 in enumerate(Y):
				x1 = c1[:,0].reshape(len(c1[:,0]),1)
				x2 = c2[:,0].reshape(len(c2[:,0]),1)
				y1 = c1[:,1].reshape(len(c1[:,1]),1)
				y2 = c2[:,1].reshape(len(c2[:,1]),1)

				# K[i][j] = pathKernel(x1,x2)
				K[i][j] = pathKernel(x1,x2) + pathKernel(y1,y2)
			print '\n'
	return K

# Takes in an array of sequences X, builds the kernel matrix s.t K_ij = K(s_i,s_j)
def frontKernelMatrix(X,Y=-1):
	if Y == -1:
		Y = X
	print len(X)
	K = np.zeros((len(X),len(Y)))
	for i, x1 in enumerate(X):
		print '***********'
		print i, '/', len(X)
		for j, x2 in enumerate(Y):
			x1 = x1.reshape(len(x1),1)
			x2 = x2.reshape(len(x2),1)
			# K[i][j] = pathKernel(x1,x2)
			K[i][j] = frontKernel(x1,x2)
		print '\n'
	return K

# Takes in an array of sequences X, builds the kernel matrix s.t K_ij = K(s_i,s_j)
def pathKernelMatrix(X):
	print len(X)
	K = np.zeros((len(X),len(X)))
	for i, x1 in enumerate(X):
		print '***********'
		print i, '/', len(X)
		for j, x2 in enumerate(X):
			x1 = x1.reshape(len(x1),1)
			x2 = x2.reshape(len(x2),1)
			# K[i][j] = pathKernel(x1,x2)
			K[i][j] = pathKernel(x1,x2)
		print '\n'
	return K

# Perform PCA on the front kernel
def kernelPCA(X, y, n_components):

	K = pathKernelMatrix(X)

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

# Time various classifications
def timing():
	# Time path kernel vs front kernel up to 120 symbols length, plot
	xvals = range(10,60,10)
	pathvals = []
	frontvals = []

	# X, y = readCoordData('../libra_data')
	# print 'Dataset size ', X.shape
	#
	#
	# clf = SVC(kernel='precomputed')
	# pathvals2 = []
	# frontvals2 = []
	#
	# for i in xvals:
	#
	# 	sub_X = X[:i]
	# 	start = time.time()
	# 	kmatrix = twoDimFrontMatrix(sub_X)
	# 	end = time.time()
	# 	timetaken = (end-start)
	# 	pathvals2.append(timetaken)
	# 	print 'Regular path kernel time (s): ', timetaken
	#
	# 	start1 = time.time()
	# 	kmatrix = twoDimPathMatrix(sub_X)
	# 	end1 = time.time()
	# 	timetaken1 = (end1-start1)
	# 	frontvals2.append(timetaken1)
	# 	print '0.05-front kernel time (s):', timetaken1
	# 	print '*---------*'
	#
	# plt.figure(figsize=(8,6))
	# plt.scatter(xvals, pathvals2, color='red', alpha=0.5, label='path kernel')
	# plt.scatter(xvals, frontvals2, color='blue', alpha=0.5, label='0.05-front kernel')
	# plt.legend(loc='upper center', shadow=True)
	#
	# plt.title('Time taken by path kernel and front kernel on subset of libras dataset')
	# plt.xlabel('Num samples')
	# plt.ylabel('Log of time taken (s)')

	for i in xvals:
		x_sing = np.linspace(0,2*pi,i)
		# y_sing = np.linspace(0,2*pi,10)
		x_sing = x_sing.reshape(i,1)
		cos_sing = np.array(np.cos(x_sing))
		sin_sing = np.array(np.sin(x_sing))
		cos_sing = cos_sing.reshape(i,1)
		sin_sing = sin_sing.reshape(i,1)

		# Time functions
		print 'For sequence length: ', i
		start = time.time()
		pathKernel( sin_sing, cos_sing )
		end = time.time()
		timetaken = (end-start)*1000
		pathvals.append(timetaken)
		print 'Regular path kernel time (ms): ', timetaken
		start1 = time.time()
		frontKernel( sin_sing, cos_sing )
		end1 = time.time()
		timetaken1 = (end1-start1)*1000
		frontvals.append(timetaken1)
		print '0.05-front kernel time (ms):', timetaken1
		print '*---------*'

	plt.figure(figsize=(8,6))
	plt.scatter(xvals, pathvals, color='red', alpha=0.5, label='path kernel')
	plt.scatter(xvals, frontvals, color='blue', alpha=0.5, label='0.05-front kernel')
	plt.legend(loc='upper center', shadow=True)

	plt.title('Time taken by front kernel and path kernel on cosine sequences')
	plt.xlabel('Sequence length')
	plt.ylabel('Log of time taken (ms)')
	plt.show()

# Read files in from a folder hierarchy, builds array of samples along with their class names
def readFolderData(folder):
	dataset = []
	folder_list = os.listdir(folder)
	for subfolder in folder_list:
		file_list = os.listdir(folder + '/' + subfolder)
		subdata = []
		for a_file in file_list:
		# for i in range(9):
			sequence = []
			f = io.open(folder + '/' + subfolder + '/' + a_file, 'r', encoding='utf-8')
			for line in f:
				vals = line.split()
				sequence.append([float(x) for x in vals])
			sample = np.array(sequence)
			name = os.path.splitext(os.path.basename(a_file))[0]
			name = name.split('-')[0]
			subdata.append({'sample':sample, 'class':name})
			f.close()
		dataset.append(subdata)

	return dataset

# Read synthetic control data
def readSynthData():
	classes = ['normal','cyclic','increasing','decreasing','upward','downward']
	dataset = []
	f = io.open('../synthcontrol/synthetic_control.data', 'r', encoding='utf-8')
	for i,line in enumerate(f):
		vals = line.split()
		sample = np.array([float(x) for x in vals])
		dataset.append({'sample':sample, 'class':classes[i/100]})
	f.close()

	return dataset

# Read files in and arrange as arrays of co-ordinates
def readCoordData(folder):
	dataset = []
	file_list = os.listdir(folder)
	for a_file in file_list:
		f = io.open(folder + '/' + a_file, 'r', encoding='utf-8')
		data = np.loadtxt(f, delimiter=",")
		dataset.append(data)
		f.close()

	# print '**', len(dataset)
	dataset = np.concatenate( dataset, axis=0 )

	# for i, data in enumerate(dataset):
	# 	X_tmp = dataset[i][:,0::2]
	# 	Y_tmp = dataset[i][:,1::2]
	X = dataset[:,0::2]
	Y = dataset[:,1::2]

	# Class labels are last elements of the X coord array
	# Stack x and y coord arrays into one array, containing (x,y) coords as array
	labels = X[:,-1]
	X = np.delete(X, np.s_[-1:], 1)
	X = np.stack((X,Y), axis=-1)

	return X, labels

# Read in libra data, format and classify
def libraClassification():
	print '--- Classifying with Libra Dataset ---'
	#** ---- Libra dataset ---- **#
	X, y = readCoordData('../libra_data')
	print 'Dataset size ', X.shape

	# -------------------------------------------- #
	## 10 fold cross validation scores

	clf = SVC(kernel='precomputed')

	print 'Building kernel matrix...'
	# sub_X = X[:50]
	# start = time.timeit()
	kmatrix = twoDimFrontMatrix(sub_X)
	# end = time.time()
	# timetaken = (end-start)*1000

	# start = time.timeit()
	# kmatrix = twoDimPathMatrixtMatrix(sub_X)
	# end = time.time()
	# timetaken = (end-start)*1000

	print 'Scoring classifier...'
	# print kmatrix.shape
	scores = cross_val_score(clf, kmatrix, y, cv=10)
	print 'cross val score: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

	# -------------------------------------------- #
	## Traint-test split of data

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	# print X_train.shape, '  ', X_test.shape

	# kmatrix_train = twoDimFrontMatrix(X_train)
	# kmatrix_test = twoDimFrontMatrix(X_test, X_train)
	# print kmatrix_test.shape
	# clf.fit(kmatrix_train, y_train)
	# y_pred = clf.predict(kmatrix_test)
	# print 'accuracy score: %0.5f' % accuracy_score(y_test, y_pred)

	# twoclasses = X[:25, :]
	# twolabels = y[:25]
	#
	# kernelPCA(twoclasses, twolabels, 2)

	# ------- --- ----- --- ------- #

# Read in synthetic control dataset, classify
def synthcontrolClassification():
	print '--- Classifying with syntehtic control dataset ---'
	data = readSynthData()

	data = DataFrame(data)
	data_samples = data['sample'].values
	data_targets = data['class'].values

	clf = SVC(kernel='precomputed')

	print 'Building kernel matrix...'
	kmatrix = frontKernelMatrix(data_samples)

	print 'Scoring classifier...'
	# print kmatrix.shape
	scores = cross_val_score(clf, kmatrix, data_targets, cv=6)
	print 'cross val score: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2)

# normalise an array
def normalize(v):
	norm=np.linalg.norm(v, ord=np.inf)
	if norm==0:
		norm=np.finfo(v.dtype).eps
	return v/norm

if __name__ == "__main__":

	###### generate some sequential datas
	pi = np.pi

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
		size = 200
		# size = random.random_integers(400,600)
		x = np.linspace(0,2*pi,size)
		x = x.reshape(size,1)
		cosx = np.array(np.cos(x))
		cosx = cosx.reshape(size,1)
		noise = random.normal(0, 0.25, size)
		noise = cosx + noise
		cosines.append(noise)
		labels.append(0)

	for i in range(10,20):
		size = 200
		# size = random.random_integers(400,600)
		x = np.linspace(0,2*pi,size)
		x = x.reshape(size,1)
		sinx = np.array(np.sin(x))
		sinx = sinx.reshape(size,1)
		noise = np.random.normal(0, 0.25, size)
		noise = sinx + noise
		cosines.append(noise)
		labels.append(1)

	#------------------------------#
	####### Run various functions

	# libraClassification()

	synthcontrolClassification()

	# k_struc = strucKernel(0.37,0.37,40,40)
	# plt.matshow(k_struc)
	# plt.show()
	# print(k_len)
	# result = pathKernel(cosx,sinx)
	# result2 = pathKernel(sinx,cosinx,kernRBF,kernExp)
	# cosines = np.array(cosines)
	# print(cosines.shape)

	# timing()
	# kernelPCA(cosines, labels, 2)
	# print('kernel matrix: ', kmatrix)
	# print('Cos and sin: ', result)
	# print(result2)

	# xvals = range(10,50,10)
	# pathvals = [210.28,810.84,1818.31,3361.07]
	# frontvals = [1.69,6.07,13.4,23.74]
	# plt.figure(figsize=(8,6))
	# plt.scatter(xvals, pathvals, color='red', alpha=0.5, label='path kernel')
	# plt.scatter(xvals, frontvals, color='blue', alpha=0.5, label='0.05-front kernel')
	# plt.legend(loc='upper center', shadow=True)
	#
	# plt.title('Time taken by front kernel and path kernel on Libras data')
	# plt.xlabel('Sequence length')
	# plt.ylabel('Time taken (s)')
	# plt.show()

sys.exit()
