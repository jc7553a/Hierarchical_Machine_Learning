import threading
import numpy as np
import Autoencoder as AE
import random as ra

global data, testing
global total_reconstruction, networks

def normalize(dataPassed):
    #global data
    #print(np.shape(data))
    'Normalize Data by Y-Min/Max-min'''
    #train = np.array(train)
    #shape = np.shape(train
    k = 0
    
    while k <len(dataPassed[0])-1:
        maxim = 0
        minim = 0
        for i in range(len(dataPassed)):
            if dataPassed[i][k] > maxim:
                maxim = dataPassed[i][k]
            if dataPassed[i][k] < minim:
                minim = dataPassed[i][k]
        denom = maxim - minim
        if denom == 0:
            denom = .0000000001
        for t in range(len(dataPassed)):
            dataPassed[t][k] = (dataPassed[t][k] - minim)/denom
        k +=1
    return dataPassed

def findSmallest(vector):
    smallest = vector[0]*1000
    sIndex = 0
    for i in range(len(vector)):
        if vector[i]*1000 < smallest:
            smallest = vector[i]*1000
            sIndex = i
    return sIndex

def getdata():
    global data, testing
    init_data =[]
    for line in open('covtype.data').readlines():
        holder = line.split(',')
        i = 0
        for i in range (len(holder)):
            holder[i] = float(holder[i])
        init_data.append(holder)
    
    new_data = normalize(init_data)
    
    del init_data
    half = int(len(new_data)*.33)
    training = new_data[0:half][:]
    testing = new_data[half:len(new_data)][:]
    end = len(new_data[0][:])-1
    setOfClasses = set()
    for i in range(len(training)):
        setOfClasses.add(training[i][end])
    numClasses = len(setOfClasses)
    del setOfClasses
    i = 1
    seperatedMatrix = []
    while i <= numClasses:
        hMatrix = []
        j = 0
        while j <(len(training)):
            if training[j][end] == i:
                hMatrix.append(training[j][:])
                del training[j]
            else:
                j+=1
        seperatedMatrix.append(hMatrix)
        i+=1
    return seperatedMatrix



def test(autoencoder):
    global testing
    r_errors = []
    for i in range(len(testing)):
        r_errors.append(autoencoder.calc_total_cost([testing[i][0:len(testing[i])-1]]))
    return r_errors
                                    

def train(dataPassed):
    print("Thread " + t.getName() + " is starting Training")
    n_features = len(dataPassed[0])-1
    network = AE.Autoencoder(n_features, int(n_features*.75))
    n_epochs = 10
    dataPassed = np.array(dataPassed)
    for i in range(n_epochs*20):
        for j in range(len(dataPassed)):
            rand = ra.randint(0, len(dataPassed)-26)
            network.partial_fit(dataPassed[rand:rand+25, 0:54])
    for i in range(n_epochs):
        for j in range(len(dataPassed)):
            rand = ra.randint(0, len(dataPassed)-1)
            network.partial_fit([dataPassed[rand][0:n_features]])
    return network

def worker():
    global data, total_reconstruction, networks
    threadVal = t.getName()
    network = train(data[int(t.getName())])
    networks[threadVal] = network
    print("Thread " + str(threadVal) + " is done")
    #reconstruction_errors = test(network)
    #total_reconstruction.append(reconstruction_errors)
    return

def initializeGlobals():
    global total_reconstruction, data, testing, networks
    total_reconstruction = []
    data = []
    testing = []
    networks = {}
        
    
if __name__ == '__main__':
    initializeGlobals()
    global data, testing
    data = getdata()
    data = np.array(data)
    
    
    threads = []
    for i in range(len(data)):
        t = threading.Thread(target = worker, name = str(i))
        threads.append(t)
        t.start()
    for thread in threads:
        thread.join()
    del data
    for i in range(len(networks)):
        total_reconstruction.append(test(networks[str(i)]))
        print("Done Testing with AE " + str(i))
    
    classVals = []
    for i in range(len(total_reconstruction[0])):
        values = []
        for j in range(len(networks)):
            values.append(total_reconstruction[j][i])
        smallest = findSmallest(values)
        classVals.append(smallest +1)
        if i < 10:
            print(values)
            print(smallest+1)
        i+=1
    correct = 0
    incorrect = 0
    for i in range(len(classVals)):
        if classVals[i] == testing[i][len(testing[0])-1]:
            correct += 1
        else:
            incorrect +=1
    print("Accuracy " + str(correct/len(testing)))
    print(total_reconstruction[0][0:10])
    
        
        
        
        
