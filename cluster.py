import numpy as np
import Autoencoder as ae
import random as ra
import matplotlib.pyplot as plt
import math
import AutoencoderChildren as aeChild
import getData as data_collection
import testingFunctions as test_func

def check_percentage_difference(array):
    minimum = min(array)
    maximum = max(array)
    difference = ((abs(maximum-minimum))/((maximum+minimum)/2))*100
    return difference


def train(data):
    n_features = len(data[0])-1
    network = ae.Autoencoder(n_features, int(n_features*.75))
    losses = []
    for i in range(25):
        midLosses = []
        for j in range(len(data)):
            rand = ra.randint(0, len(data))
            midLosses.append(network.partial_fit([data[rand][0:len(data[rand])-1]]))
        losses.append(np.average(midLosses))
    del losses
    return network

def test(auto, data):
    res = []
    half = int(len(data)/2)
    for i in range(int(len(data)/2)):
        res.append(auto.calc_total_cost([data[i+half][0:len(data[i])-1]]))
    return res

def onlineTrain(data, network):
    start = int(len(data)/2)
    losses = []
    for i in range(start):
        losses.append(network.partial_fit([data[i+start][0:len(data[i])-1]]))

    #plt.plot(losses)
    #plt.show()
    #splits = askQuestions(losses)
    return splits
                

def childrenTrain(network, children, data):
    print("Got To Gating")
    print(children[len(children)-1].getThresholdHigh())
    print(children[len(children)-1].getThresholdLow())
    for j in range(1):
        for i in range(int(len(data)/2)):
            rand = ra.randint(0, int(len(data/2))-1)
            re = network.calc_total_cost([data[rand][0:len(data[rand])-1]])
            for k in range(len(children)):
                child = children[k]
                if child.getThresholdHigh() > re and child.getThresholdLow() < re:
                    child_in_use = children[k]
            child_in_use.partial_fit(network.reconstruct([data[rand][0:len(data[rand])-1]]))
    return network

def doChildren(network, data, splits):
    n_features = len(data[0])-1
    for i in range(len(splits)):
        if i == 0:
            network.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), splits[i], 0))
        else:
            network.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), splits[i], split[i-1]))
    network.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), 120, splits[len(splits)-1]))
    children = network.getChildren()
    network = childrenTrain(network, children, data)
    return network

def testTree(network, data):
    half = int(len(data)/2)
    losses = []
    for i in range(half):
        myData = data[i][0:len(data[i])-1]
        losses.append(jp.climbTree(network, myData))
    plt.plot(losses)
    plt.show()



if __name__ == '__main__':
    '''Data Collection'''
    data = data_collection.importData('covtype_norm.csv')
    data_length = len(data)
    training_length = int(len(data)*.25)
    temp_training_data = data[0:training_length][:]
    training_data = data_collection.splitTrainingData(temp_training_data, 1)
    '''Garbage Cleanup'''
    del temp_training_data
    del data

    '''Start Process of Training and Creating Hierarchical Tree'''
    network = train(training_data)
    
    testTree(network, data)
    #err = test(network, newData)
    #splits = askQuestions(err)

    
    
