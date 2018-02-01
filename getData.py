import numpy as np
import os


def normalize(train):
    'Normalize Data by Y-Min/Max-min'''
    train = np.array(train)
    shape = np.shape(train)
    k = 0
    while k <shape[1]-1:
        maxim = 0
        minim = 0
        for i in range(shape[0]):
            if train[i][k] > maxim:
                maxim = train[i][k]
            if train[i][k] < minim:
                minim = train[i][k]
        denom = maxim - minim
        if denom == 0:
            denom = .0000000001
        for t in range(shape[0]):
            train[t][k] = (train[t][k] - minim)/denom
        k +=1
    return train

def splitTrainingData(data, classification):
    training_data = []
    class_val = len(data[0])-1
    for i in range(len(data)):
        if data[i][class_val] == classification:
            training_data.append(data[i][:])
    return training_data

def importData(file):
    os.chdir('C:/Data_Sets')
    data = np.genfromtxt(file, delimiter = ',')
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    return data
