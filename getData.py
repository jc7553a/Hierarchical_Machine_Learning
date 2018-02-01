import numpy as np


def normalize(train):
    'Normalize Data by Y-Min/Max-min'''
    train = np.array(train)
    shape = np.shape(train)
    k = 0
    while k <shape[1]:
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

def importData(file):
    data = np.genfromtxt(file, delimiter = ',')
    newData = []
    for i in range(len(data)):
        holder = []
        for j in range(len(data[0])):
            if j != 1:
                holder.append(float(data[i][j]))
            newData.append(holder)
    newData = normalize(newData)
    return newData
