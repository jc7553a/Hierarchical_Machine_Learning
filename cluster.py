import numpy as np
import Autoencoder as ae
import random as ra
import matplotlib.pyplot as plt
import math
import AutoencoderChildren as aeChild
import getData as data_collection
import testingFunctions as test_func
import binning

def check_percentage_difference(array):
    minimum = min(array)
    maximum = max(array)
    difference = ((abs(maximum-minimum))/((maximum+minimum)/2))*100
    return difference


def train(data):
    n_features = len(data[0])-1
    network = ae.Autoencoder(n_features, int(n_features*.75))
    losses = []
    '''Start Training Initial Autoencoder for some time'''
    print("Training Initial Autoencoder")
    for i in range(25):
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            network.partial_fit([data[rand][0:n_features]])
    print("Checking Loss Values of Initial Autoencoder")
    for i in range(len(data)):
        losses.append(network.calc_total_cost([data[i][0:n_features]]))
    print("checking perctage difference")
    got_bins = False
    if check_percentage_difference(losses) > 150:
        bins = binning.binning(losses)
        got_bins = True
        print("Bins " + str(len(bins)))
    print("Percentage Difference " + str(check_percentage_difference(losses)))
    if got_bins:
        for i in range(len(bins)):
            network.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), max(bins[i]), min(bins[i])))
    network = childrenTrain(network, network.getChildren(), data)
        
    return network

def test(auto, data):
    res = []
    half = int(len(data)/2)
    for i in range(int(len(data)/2)):
        res.append(auto.calc_total_cost([data[i+half][0:len(data[i])-1]]))
    return res

                

def childrenTrain(network, children, data):
    print("Training Children")
    print(children[len(children)-1].getThresholdHigh())
    print(children[len(children)-1].getThresholdLow())
    for j in range(25):
        for i in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            re = network.calc_total_cost([data[rand][0:len(data[rand])-1]])
            for k in range(len(children)):
                child = children[k]
                if child.getThresholdHigh() > re and child.getThresholdLow() < re:
                    child_in_use = children[k]
            child_in_use.partial_fit(network.reconstruct([data[rand][0:len(data[rand])-1]]))
    return network


def testTree(network, data):
    print("Testing The Tree Network")
    losses = []
    for i in range(len(data)):
        myData = data[i][0:len(data[i])-1]
        losses.append(test_func.climbTree(network, myData))

    return losses

def set_threshold(network, testing):
    print("Setting Threshold")
    losses = []
    for i in range(len(testing)):
        data_pass = testing[i][0:len(testing[i])-1]
        losses.append(test_func.climbTree(network, data_pass))
    threshold = np.average(losses) + (3*(np.std(losses)))
    return threshold
        



if __name__ == '__main__':
    '''Data Collection'''
    data = data_collection.importData('covtype_norm.csv')
    data_length = len(data)
    training_length = int(len(data)*.25)
    temp_training_data = data[0:training_length][:]
    training_data = data_collection.splitTrainingData(temp_training_data, 1)
    testing_data = data[training_length:len(data)][:]
    
    '''Garbage Cleanup'''
    del temp_training_data
    del data

    '''Start Process of Training and Creating Hierarchical Tree'''
    network = train(training_data)

    '''Create a Threshold'''
    threshold = set_threshold(network, testing_data[0:100][:])

    '''Test our 2 Layer Tree Structure'''
    re = testTree(network, testing_data )
    print("Starting to Look at Predicted Values")
    predicted_values = []
    for i in range(len(re)):
        if re[i] < threshold:
            predicted_values.append(1)
        else:
            predicted_values.append(0)

    correct = 0
    incorrect = 0
    for i in range(len(testing_data)):
        if predicted_values[i] == 1 and  testing_data[i][len(testing_data[i])-1] == 1:
            correct +=1
        if predicted_values[i] == 0 and testing_data[i][len(testing_data[i])-1] != 1:
            correct +=1
        if predicted_values[i] == 1 and testing_data[i][len(testing_data[i])-1] != 1:
            incorrect +=1
        if predicted_values[i] == 0 and testing_data[i][len(testing_data[i])-1] == 1:
            incorrect +=1
    print("Correct " + str(correct))
    print("Incorrect " + str(incorrect))
    print("Accuracy " + str(float(correct/(len(testing_data)))))

    
    
