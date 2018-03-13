# -*- coding: utf-8 -*-
"""
Case 1

@author: Martin Meincke
"""
import numpy as np
import pandas as pd


path = ""
dataPath = path + 'data.csv'
attributeNames = []

# Dump data file into an array
with open(dataPath, "r") as ins:
    listArray = []
    for line in ins:
        # Remove junk, irritating formating stuff
        listArray.append(line.replace('\n', '').split('\t'))


n = len(listArray) - 1
p = len(listArray[0][0].split(','))
x100 = []

for i, data in enumerate(listArray):
    dataTemp = data[0].split(',')
    y = dataTemp[0]
    x = dataTemp[1:p]

    if i == 0: # first row is attribute names
        attributeNames = dataTemp[0:n]
    else:
    	x100.append(dataTemp[p-1])

x100Names = sorted(set(x100))
x100Dict = dict(zip(x100Names,range(len(x100Names))))

p_transformed = p + len(x100Names) -  2
X_test = np.zeros((100, p_transformed))
Y_test = np.zeros((100,1))
X_train = np.zeros((1000, p_transformed))

for i, data in enumerate(listArray):
	dataTemp = data[0].split(',')
	x = dataTemp[1:p-1]
	if i > 0 and i <= 100:
		x100_one_hot = np.zeros((len(x100Names),1))
		x100_one_hot[x100Dict.get(x100[i])] = 1

		X_test[i-1] = np.append(x,x100_one_hot.T[0])
		Y_test[i-1] = dataTemp[0]
	else:
		pass #print(i)

print(Y_test)


