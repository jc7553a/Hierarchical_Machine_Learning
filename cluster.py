import numpy as np
import Autoencoder as ae
import random as ra
import matplotlib.pyplot as plt
import math
import AutoencoderChildren as aeChild
from getData import *
import testingFunctions as jp


def train(data):
    n_features = len(data[0])-1
    network = ae.Autoencoder(n_features, int(n_features*.75))
    losses = []
    for i in range(1):
        midLosses = []
        for j in range(int(len(data)/2)):
            rand = ra.randint(0, int(len(data)/2))
            midLosses.append(network.partial_fit([data[rand][0:len(data[rand])-1]]))
        losses.append(np.average(midLosses))
    #plt.plot(losses)
    #plt.show()
    del losses
    return network

def test(auto, data):
    res = []
    half = int(len(data)/2)
    for i in range(int(len(data)/2)):
        res.append(auto.calc_total_cost([data[i+half][0:len(data[i])-1]]))
    return res

def split(res, splits):
    resMatrix = []
    for i in range(len(res)):
        holder = []
        holder.append(res[i])
        holder.append(i)
        resMatrix.append(holder)
    resMatrix = np.array(resMatrix)
    resMatrix[resMatrix[:,1].argsort()]
    returnTensor = []
    for i in range(len(splits)):
        holder = []
        j = 0
        while j <(len(res)):
            if res[j][0] < splits[i]:
                holder.append(res[j])
                del res[j]
                j -=1
            j+=1
        returnTensor.append(holder)
    return returnTensor
def onlineTrain(data, network):
    start = int(len(data)/2)
    losses = []
    for i in range(start):
        losses.append(network.partial_fit([data[i+start][0:len(data[i])-1]]))

    #plt.plot(losses)
    #plt.show()
    splits = askQuestions(losses)
    return splits
                

def askQuestions(errors):
    print("Do you want a visual graph? Type 0 for yes")
    graph = int(input())
    if graph == 0:
        plt.plot(errors)
        plt.show()
    print("How many splits do you see?")
    splits = input()
    splits = int(splits)
    splitsVals = []
    for i in range(splits):
        print("Where does Split " + str(i+1) + " begin?")
        splitsVals.append(float(input()))
    return splitsVals

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
    data = importData('benignCancerNew.txt')
    network = train(data)
    split = onlineTrain(data, network)
    if len(split) == 0:
        print ("Done")
    else:
        network = doChildren(network, data, split)
    
    testTree(network, data)
    #err = test(network, newData)
    #splits = askQuestions(err)

    
    
