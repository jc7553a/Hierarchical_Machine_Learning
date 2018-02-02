def binning(array):
    minimum = min(array)
    maximum = max(array)
    percentage = (maximum - minimum)*.1
    bins = []
    for j in range(10):
        holding_bin = []
        upper_percentage = percentage*(j+1)
        lower_percentage = percentage*(j)
        for i in range(len(array)):
            if array[i] <= (minimum + upper_percentage) and array[i] >= (minimum +lower_percentage):
                holding_bin.append(array[i])
        bins.append(holding_bin)
    bins2 = [x for x in bins if x != []]
    return merging_bins(bins2, 100)


def merging_bins(bins, min_bin):
    merged_bins = []
    i= 0
    while i <(len(bins)-1):
        if len(bins[i]) < min_bin  or len(bins[i+1]) < min_bin:
            bins[i] = bins[i] + bins[i+1]
            del bins[i+1]
            i-=1
        i +=1
    for i in range(len(bins)):
        bins[i] = sorted(bins[i])
    return bins
    


