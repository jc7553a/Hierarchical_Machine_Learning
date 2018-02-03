'''Train the Root Initially with this function'''
def initial_train(data):
    n_features = len(data[0])-1
    network = ae.Autoencoder(n_features, int(n_features*.75))
    n_epochs = 25
    for i in range(n_epochs):
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            network.partial_fit([data[rand][0:n_features]])
    return network

'''Helper Function to Find Child of Node that gets the data instance'''
def findChild(children, re):
    for i in range(len(children)):
        child = children[i]
        if re > child.getThresholdLow() and re < child.getThresholdHigh():
            return child
    return children[len(children)-1]


'''After Root is Trained, Apply training to tree through traversal'''
def traverse_train(root, reconstructed, re):
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
    
    


def train(data):
    root = initial_train(data)
            
