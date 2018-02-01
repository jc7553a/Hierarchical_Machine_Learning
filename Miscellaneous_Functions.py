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
