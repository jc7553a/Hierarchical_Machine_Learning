import random as ra
import AutoencoderChildren as aeChild
import binning
import Autoencoder as ae


'''global variable'''
done_training = True
levels_created = 0
n_epochs = 1

'''Get percentage distance of smallest reconstruction error to largest'''
def check_percentage_difference(array):
    print("Checking Percentage Difference")
    minimum = min(array)
    maximum = max(array)
    difference = ((abs(maximum-minimum))/((maximum+minimum)/2))*100
    print("Difference : " + str(difference))
    return difference

'''Helper Function to Find Child of Node that gets the data instance'''
def findChild(children, re):
    for i in range(len(children)):
        child = children[i]
        if re > child.getThresholdLow() and re < child.getThresholdHigh():
            return child
    return children[len(children)-1]



'''Train the Root Initially with this function'''
def initial_train(data):
    global n_epochs
    print("Start Training Root Node")
    n_features = len(data[0])-1
    network = ae.Autoencoder(n_features, int(n_features*.75))
    for i in range(n_epochs):
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            network.partial_fit([data[rand][0:n_features]])
    losses = []
    for i in range(len(data)):
        losses.append(network.calc_total_cost([data[i][0:n_features]]))
    if check_percentage_difference(losses) > 150:
        bins = binning.binning(losses)
        if len(bins) > 1:
            for i in range(len(bins)):
                network.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), max(bins[i]),min(bins[i])))
    return network


'''After Root is Trained, Apply training to tree through traversal'''
def traverse_train(root, reconstructed, re):
    temp = root
    children = temp.getChildren()
    done  = True
    while (done):
        children = temp.getChildren()
        if (len(children) == 0):
            temp.partial_fit(reconstructed)
            done = False
        else:
            child = findChild(children, re)
            temp = child
            re = temp.calc_total_cost(reconstructed)
            reconstructed = temp.reconstruct(reconstructed)

                    
''' After Training Do an Epoch of Tests'''
def traverse_test(root, data):
    re = root.calc_total_cost([data])
    reconstructed = root.reconstruct([data])
    temp = root
    children = temp.getChildren()
    done  = True
    while (done):
        children = temp.getChildren()
        if (len(children) == 0):
            child_re = temp.calc_total_cost(reconstructed)
            temp.addLoss(child_re)
            done = False
        else:
            child = findChild(children, re)
            temp = child
            re = temp.calc_total_cost(reconstructed)
            reconstructed = temp.reconstruct(reconstructed)

def check_for_splitting(root, n_features):
    global done_training
    print("In splitting " + str(len(root.getChildren())))
    children = root.getChildren()
    for i in range (len(children)):
        child = children[i]
        if (len(child.getChildren())) == 0:
            print("Here")
            #print(child.getLosses())
            if len(child.getLosses()) > 0 and check_percentage_difference(child.getLosses()) > 15:
                #print("Farts")
                bins= binning.binning(child.getLosses())
                done_training = False
                for i in range(len(bins)):
                    child.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), max(bins[i]),min(bins[i])))
                    child.cleanUpLoss()
        else:
            print(len(child.getLosses()))
            check_for_splitting(child, n_features)
                    
'''After done training we will set all children to done trained'''
def set_tree_network_to_trained(network):
    for net in network.getChildren():
        net.setTrained(True)
        set_tree_network_to_trained(net)

'''checking to see if there are any nodes that haven't been trained'''
def test_done_training(root):
    global done_training
    for child in root.getChildren():
        if child.getTrained() == False:
            done_testing = False

'''Recursively Train Tree until No more Creation of Levels to be done'''
'''Or I set levels_created to 4 and then exit for now'''
def train_tree(root, data):
    global done_testing, levels_created, n_epochs
    print("Training Tree Levels = " + str(levels_created))
    print("Checking root In Train Tree Id: " + str(root.getId()))
    done_testing = True
    if levels_created == 3:
        return root
    n_features = len(data[0])-1
    '''Train the Tree'''
    for i in range(n_epochs):
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            re = root.calc_total_cost([data[rand][0:n_features]])
            reconstructed = root.reconstruct([data[rand][0:n_features]])
            traverse_train(root, reconstructed, re)
    print("Done Training Setting Nodes to Trained")
    set_tree_network_to_trained(root)             #Set all Nodes to Trained
    for i in range(len(data)):
        traverse_test(root, data[i][0:n_features])                     #Get Losses on 1 Epoch for all Children that are Leaves
    print("Checking for Splits")
    check_for_splitting(root, n_features)         #See if any node needs to Split
    #test_done_training(root)                     #See if Any node hasn't been Trained (Probably Don't need this)
    levels_created +=1                            #Add 1 to level Created So we  have atleast an Exit point eventually
    if done_training == True:                      #If all Nodes are Trained and Pass my splitting test, return the root
        return root
    else:
        print("There was a split")
        train_tree(root, data)                    #Otherwise some node hasn't been trained and must now train it, through gating process
    
def train(data):
    print("Started Training")
    root = initial_train(data)                    #Intially Train the Root Node
    if len(root.getChildren()) == 0:
        return root
    else:
        print("Checking Root Begin Id: " +str(root.getId()))
        train_tree(root, data)                 #Then Recursively Train Tree Network and add nodes until it passes the test
        return root
