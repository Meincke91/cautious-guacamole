# -*- coding: utf-8 -*-
"""
Case 1

@author: Martin Meincke
"""
import numpy as np
import pandas as pd
import math

class data_clean():
    """docstring for ClassName"""
    def __init__(self, attribute_names=[]):
        super(data_clean, self).__init__()
        self.attribute_names = attribute_names
        
    def get_data_from_file(self, path):
        # Dump data file into an array
        listArray = []

        with open(path, "r") as ins:
            for line in ins:
                # Remove junk, irritating formating stuff
                listArray.append(line.replace('\n', '').split('\t'))
        return listArray

    def get_test_and_train(self, data_list):
        n = len(data_list) - 1
        p = len(data_list[0][0].split(','))
        x100 = []

        for i, data in enumerate(data_list):
            dataTemp = data[0].split(',')
            y = dataTemp[0]
            x = dataTemp[1:p]

            if i == 0: # first row is attribute names
                self.attribute_names = dataTemp[0:n]
            else:
                x100.append(dataTemp[p-1])

        x100Names = sorted(set(x100))
        x100Dict = dict(zip(x100Names,range(len(x100Names))))

        p_transformed = p + len(x100Names) -  2
        X_test = np.zeros((100, p_transformed))
        Y_test = np.zeros((100,1))

        X_train = np.zeros((len(data_list)-len(X_test[:,0])-1, p_transformed))

        for i, data in enumerate(data_list):
            dataTemp = data[0].split(',')
            x = dataTemp[1:p-1]
            if i > 0 and i <= 100:
                x100_one_hot = np.zeros((len(x100Names),1))
                x100_one_hot[x100Dict.get(x100[i])] = 1

                X_test[i-1] = np.append(x,x100_one_hot.T[0])
                Y_test[i-1] = dataTemp[0]
            elif i > 100:
                x100_one_hot = np.zeros((len(x100Names),1))
                x100_one_hot[x100Dict.get(x100[i-1])] = 1
                X_train[i-101] = np.append(x,x100_one_hot.T[0])

        return X_test, Y_test, X_train

    def replace_NaN_with_mean(self, column):
        sum_of_valid_data = 0
        count_of_valid = 0

        for data in column:
            if not math.isnan(data):
                print(data)
                sum_of_valid_data = sum_of_valid_data + data
                count_of_valid += 1

        print(sum_of_valid_data)
        print(sum_of_valid_data/count_of_valid)



