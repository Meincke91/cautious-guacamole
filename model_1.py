# -*- coding: utf-8 -*-
"""
Case 1

@author: Martin Meincke
"""
import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import data_clean as dc

class model_1():
	"""docstring for model_1"""
	def __init__(self):
		super(model_1, self).__init__()
		self.dc = dc.data_clean()

	def train_model(self):
		data = self.dc.get_data_from_file(path="data.csv")
		X_test, Y_test, X_train = self.dc.get_test_and_train(data_list=data)

		self.dc.replace_NaN_with_mean(X_train[0,:])

		#print(normalize(X_train[np.newaxis,0], axis=0).ravel())
		
if __name__ == "__main__":
	model = model_1()
	model.train_model()
