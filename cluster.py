import numpy as np
import Autoencoder as ae
import random as ra
import matplotlib.pyplot as plt
import math
import AutoencoderChildren as aeChild
import getData as data_collection
import testingFunctions as test_func
import binning
import trainingFunctions as train



def test(auto, data):
    res = []
    half = int(len(data)/2)
    for i in range(int(len(data)/2)):
        res.append(auto.calc_total_cost([data[i+half][0:len(data[i])-1]]))
    return res

                


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
    threshold = np.average(losses) + (1*(np.std(losses)))
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
    network = train.train(training_data)

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

    
    
