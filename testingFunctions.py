def findChild(network, re, children):
    for i in range(len(children)):
        child = children[i]
        if re > child.getThresholdLow() and re < child.getThresholdHigh():
            return child
    return children[len(children)-1]

def climbTree(network, data):
    re = network.calc_total_cost([data])
    reconstruction = network.reconstruct([data])
    while (len(network.getChildren())!= 0):
        child = findChild(network, re, network.getChildren())
        re = child.calc_total_cost(reconstruction)
        reconstruction = child.reconstruct(reconstruction)
        network = child
    return re
