import random as ra
import AutoencoderChild as aeChild
import binning


'''global variable'''
done_training = True
levels_created = 0

'''Get percentage distance of smallest reconstruction error to largest'''
def check_percentage_difference(array):
    minimum = min(array)
    maximum = max(array)
    difference = ((abs(maximum-minimum))/((maximum+minimum)/2))*100
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
    n_features = len(data[0])-1
    network = ae.Autoencoder(n_features, int(n_features*.75))
    n_epochs = 25
    for i in range(n_epochs):
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            network.partial_fit([data[rand][0:n_features]])
    losses = []
    for i in range(len(data)):
        losses.append(network.calc_total_cost([data[i][0:n_features]]))
    if check_percentage_difference(losses) > 150:
        bins = binning.binning(losses)
        for i in range(len(bins)):
            network.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), max(bins[i]),min(bins[i])))
    return network


'''After Root is Trained, Apply training to tree through traversal'''
def traverse_train(root, reconstructed, re):
    temp = root
    while (len(temp.getChildren()) != 0):
                children = temp.getChildren()
                if len(children) == 0:
                    temp.partial_fit(reconstructed)
                else:
                    for k in range(len(children)):
                        child = findChild(children, re)
                    temp = child
                    re = temp.calc_total_cost(reconstructed)
                    reconstructed = temp.reconstruct(reconstructed)

                    
''' After Training Do an Epoch of Tests'''
def traverse_test(root, data):
    re = root.calc_total_cost([data])
    reconstructed = root.reconstruct([data])
    while (len(temp.getChildren()) != 0):
                children = temp.getChildren()
                if len(children) == 0:
                    child_re = temp.calc_total_cost(reconstructed)
                    temp.addLoss(child_re)
                else:
                    for k in range(len(children)):
                        child = findChild(children, re)
                    temp = child
                    re = temp.calc_total_cost(reconstructed)
                    reconstructed = temp.reconstruct(reconstructed)

def check_for_splitting(root, n_features):
    global done_training
    for child in root.getChildren():
        if (len(child.getChildren())) == 0:
            if check_percentage_difference(child.getLosses()) > 150:
                bins= binning.binning(child.getLosses())
                done_training = False
                for i in range(len(bins)):
                    child.addChild(aeChild.Autoencoder(n_features, int(n_features*.75), max(bins[i]),min(bins[i])))
                    child.cleanUpLoss()
        else:
            check_for_splitting(child, n_features)
                    
'''After done training we will set all children to done trained'''
def set_tree_network_to_trained(network):
    for net in network.getChildren():
        net.setTrained(True)
        set_tree_network_to_trained(child)

'''checking to see if there are any nodes that haven't been trained'''
def test_done_training(root):
    global done_training
    for child in root.getChildren():
        if child.getTrained() == False:
            done_testing = False

'''Recursively Train Tree until No more Creation of Levels to be done'''
'''Or I set levels_created to 4 and then exit for now'''
def train_tree(root, data):
    global done_testing, levels_created
    
    done_testing = True
    if levels_created == 4:
        return root
    
    n_epochs = 25
    n_features = len(data[0])-1
    '''Train the Tree'''
    for i in range(n_epochs):
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            re = root.calc_total_cost([data[rand][0:n_features]])
            reconstructed = root.reconstruct([data[rand][0:n_features]])
            traverse_train(root, reconstructed, re)
            
    set_tree_network_to_trained(root)             #Set all Nodes to Trained
    traverse_test(root, data)                     #Get Losses on 1 Epoch for all Children that are Leaves
    check_for_splitting(root, n_features)         #See if any node needs to Split
    #test_done_training(root)                     #See if Any node hasn't been Trained (Probably Don't need this)
    levels_created +=1                            #Add 1 to level Created So we  have atleast an Exit point eventually
    if done_testing == True:                      #If all Nodes are Trained and Pass my splitting test, return the root
        return root
    else:
        train_tree(root, data)                    #Otherwise some node hasn't been trained and must now train it, through gating process
    
def train(data):
    root = initial_train(data)                    #Intially Train the Root Node
    if len(root.getChildren()) == 0:
        return root
    else:
        root = train_tree(root, data)                 #Then Recursively Train Tree Network and add nodes until it passes the test
        return root
