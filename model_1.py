# -*- coding: utf-8 -*-
"""
Case 1

@author: Martin Meincke
"""
import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt
import data_clean as dc
import pandas as pd

from sklearn import preprocessing

from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

class model_1():
	"""docstring for model_1"""
	def __init__(self):
		super(model_1, self).__init__()
		self.dc = dc.data_clean()

	def train_model(self):
		data = self.dc.get_data_from_file(path="data.csv")
		attribute_names, X, y, X_unlabeled = self.dc.get_test_and_train(data_list=data)

		X =  self.dc.replace_all_NaN(X)
		X_unlabeled = self.dc.replace_all_NaN(X_unlabeled)

		scaler = preprocessing.StandardScaler().fit(X)
		#X = scaler.transform(X)
		plt.show()
		X = preprocessing.normalize(X)
		#X_train, X_test, y_train, y_test = train_test_split(X_test, Y_test, test_size=0.001, random_state=0)

		print(X.shape)
		print(y.shape)
		"""
		X_train = self.dc.replace_all_NaN(X_train)
		X_train = preprocessing.normalize(X_train, axis=0)

		X_test =  self.dc.replace_all_NaN(X_test)
		X_test = preprocessing.normalize(X_test, axis=0)
		scaler = preprocessing.StandardScaler().fit(X_train)

		mu, sigma = 0, 1 # mean and standard deviation
		#s = np.random.normal(mu, sigma, 1000)
		gaussian_data = np.zeros((len(X_train),len(X_train[0])))
		for i, _ in enumerate(gaussian_data):
			gaussian_data[i] = np.random.normal(mu, sigma, len(X_train[0]))
		"""

		
		#print(scaler.transform(X_train).mean())
		#print(scaler.transform(X_train).std(axis=0))

		# PCA print and plot
		'''
		pca = PCA()
		pca.fit(X, y)
		
		pca2 = PCA()
		pca2.fit(scaler.transform(preprocessing.normalize(gaussian_data, axis=0)))

		print(pca.singular_values_)
		print(pca2.singular_values_)

		plt.scatter(np.arange(0, len(pca.singular_values_) ), pca.singular_values_)
		plt.scatter(np.arange(0, len(pca2.singular_values_) ), pca2.singular_values_)
	
		plt.show()
		'''

		# simple linear fit
		lin = LinearRegression()
		#lin.fit(X_train, y_train)
		scores_lin = cross_val_score(lin, X, y, cv=99, scoring='neg_mean_squared_error')
		print("LinearRegression Accuracy: %0.2f (+/- %0.2f)" % (scores_lin.mean(), scores_lin.std() * 2))

		# Support Vector Regression
		svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
		svr_lin = SVR(kernel='linear', C=1e3)
		svr_poly = SVR(kernel='poly', C=1e3, degree=2)

		scores_svr_rbf = cross_val_score(svr_rbf , X, y.ravel(), cv=99, scoring='neg_mean_squared_error') 
		print("svr_rbf Accuracy: %0.2f (+/- %0.2f)" % (scores_svr_rbf.mean(), scores_svr_rbf.std() * 2))

		scores_svr_lin = cross_val_score(svr_lin , X, y.ravel(), cv=99, scoring='neg_mean_squared_error') 
		print("svr_lin Accuracy: %0.2f (+/- %0.2f)" % (scores_svr_lin.mean(), scores_svr_lin.std() * 2))

		scores_svr_poly = cross_val_score(svr_poly , X, y.ravel(), cv=99, scoring='neg_mean_squared_error') 
		print("svr_poly Accuracy: %0.2f (+/- %0.2f)" % (scores_svr_poly.mean(), scores_svr_poly.std() * 2))
		# Kernel Ridge Regression
		'''
		mean_errors = []
		optimal_degree = 999
		for i in range(50,150):
			krg  = KernelRidge(kernel="polynomial", degree=i, alpha=[1e0, 0.1, 1e-2, 1e-3, 1e-4])
			scores_krg = cross_val_score(krg , X, y, cv=99, scoring='neg_mean_squared_error') 
			print(i," KernelRidge Accuracy: %0.2f (+/- %0.2f)" % (scores_krg.mean(), scores_krg.std() * 2))
			
			mean_errors.append(scores_krg.mean())
			print(mean_errors[i-51] , ">", scores_krg.mean())
			if i-50 > 0 and mean_errors[i-51] < scores_krg.mean():

				optimal_degree = i

		plt.scatter([i for i in range(0,len(mean_errors))], [-1 * i for i in mean_errors])
		plt.show()
		'''
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)
		#for i in range(0,len(X)):
		#	print("max: ", np.amax(X[:,i]),", min: ",np.amin(X[:,i]),", mean: ",np.mean(X[:,i]))

		krg  = KernelRidge(kernel="laplacian", alpha=[1e0, 0.1, 1e-2, 1e-3, 1e-4])
		scores_krg = cross_val_score(krg , X, y, cv=99, scoring='neg_mean_squared_error') 
		print("KernelRidge Accuracy: %0.2f (+/- %0.2f)" % (scores_krg.mean(), scores_krg.std() * 2))

		krg.fit(X_train, y_train)
		print("predicted: ",krg.predict(X_test)," , actual: ", y_test[0])
		

		#plt.scatter(pca.transform(X)[0], y_train)
		#plt.scatter([1 for i in range(0,len(y_train))], y_train)
		#plt.show()
		


		# Various plots
		"""
		plt.plot(pca.singular_values_,np.arange(0, len(pca.singular_values_) ) )
		
		plt.plot(np.cumsum(pca.explained_variance_ratio_))
		plt.show()
		print(pca.explained_variance_ratio_) 
		df = pd.DataFrame(scaler.transform(X_train))


		plt.matshow(df.corr())
		plt.show()
		
		for i in range(10,30):
			plt.scatter(X_test[:,i], Y_test, alpha=0.5)
			plt.xlabel("x"+str(i))
			plt.ylabel('y')
			plt.show()
		
		X_scaled = preprocessing.scale(no_nan_x)
		x_normalized = preprocessing.normalize(no_nan_x, axis=0)
		print(scaler.mean_)
		print(X_scaled.std(axis=0))
		for num in X_scaled[:,0]:
			print(num)
		self.dc.replace_NaN_with_mean(X_train[:,0 ])

		print(normalize(X_train[np.newaxis,0], axis=0).ravel())
		"""
	def prediction_accuracy(self, model, x_test, y_test):
		sum_error = 0
		for i, x in enumerate(x_test):
			sum_error += np.linalg.norm(x - y_test[i])
			

if __name__ == "__main__":
	model = model_1()
	model.train_model()
