import numpy as np
from matplotlib import pyplot as plt
import csv


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT: 
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """

    bodyfat = open(filename)
    csvreader = csv.reader(bodyfat)
    header = next(csvreader)

    dataset = []
    ##dataset.append(header)

    for row in csvreader:
        row.pop(0)
        newRow = []
        for num in row:
            newRow.append(float(num))
        
        dataset.append(newRow)

        


    return np.array(dataset)


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """

    n = 0
    sampleMean = 0
    for row in dataset:
        
        n += 1
        sampleMean += row[col]
    

    sampleMean = sampleMean / n

    sampleStd = 0

    for row in dataset:
        singleTerm = (row[col] - sampleMean) ** 2
        sampleStd = sampleStd + singleTerm
        
    
    sampleStd  = (sampleStd / (n - 1)) ** .5

    ## required print statements
    print(n)
    print('{:.2f}'.format(sampleMean))
    print('{:.2f}'.format(sampleStd))







def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """

    ## column zero represents bodyfat (this is the actual value)
    mse = 0
    for row in dataset:
        ## now will calculate expected value using linear model
        expectedPercent = betas[0]
        for i in range(len(cols)):
            expectedPercent += (betas[i + 1] * row[cols[i]])
        
        actualPercent = row[0]
        ## squared err is the difference squared
        squaredError = (expectedPercent - actualPercent) ** 2
        mse += squaredError
    

    ## getting the mean squared by averaging error
    mse = mse / len(dataset)

    







    return mse

## method evaluates the difference between the regression function and the actual value on one row
def calcSingleRow(row, cols, betas):
    
    val = betas[0]
    
    for i in range(len(cols)):
        val += betas[i + 1] * row[cols[i]]
    
    val -= row[0]
    
    return val





def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """

    gradientArr = []

    ## must collect as many derivatives as there betas (one for each)
    for i in range(len(betas)):
            gradient = 0
            for row in dataset:

                ## for beta0, the calculation doesnt include multiplication of the row term
                if i == 0:
                    gradient += calcSingleRow(row, cols, betas)
                else:
                    gradient += calcSingleRow(row, cols, betas) * row[cols[i - 1]]
            
            gradientArr.append(gradient * 2 / len(dataset))

    grads = np.array(gradientArr)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """

    currBetas = betas
    
    for i in range(T):
        newBetas = []
        ## calculates new beta value using formula (based on old beta value)
        gradient = gradient_descent(dataset, cols, currBetas)
        for j in range(len(betas)):
            newBeta = currBetas[j] - (eta * gradient[j])
            newBetas.append(newBeta)
        

        ## required print statements
        print(i + 1, '{:.2f}'.format(regression(dataset, cols, newBetas)), end = " ")
        for newBeta in newBetas:
            print('{:.2f}'.format(newBeta), end = " ")
        
        print()

        currBetas = newBetas
        



def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """

    alteredSet = []

    ## to get beta0 using the matrix method, must add a column of 1s to the beginning of the dataset.
    for row in dataset:
        rowToAdd = []
        rowToAdd.append(1)
        for i in range(len(row)):
            if i in cols:
                rowToAdd.append(row[i])
        
        alteredSet.append(rowToAdd)


        
    
    X = np.array(alteredSet)
    Xt = np.transpose(X)

    firstChunk = np.dot(Xt, X)
    firstChunkInverse = np.linalg.pinv(firstChunk)



    secondChunk = np.dot(Xt, np.transpose(dataset[:, 0]))
    betas = np.dot(firstChunkInverse, secondChunk)

    mse = regression(dataset, cols, betas)

    return (mse, *betas)


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """

    ## simply using the dataset and cols to come up with the betas
    betas = []
    mseAndBetas = compute_betas(dataset, cols)
    for i in range(len(mseAndBetas)):
        if i != 0:
            betas.append(mseAndBetas[i])

    ## once we have betas, can plug in the feature values for X1, X2, ... and get the prediction based on the model
    result = betas[0]

    for i in range(len(cols)):
        result += (features[i] * betas[i + 1])

    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """

    linear = []
    quadratic = []

    

    for i in range(len(X)):
        ly = betas[0] + betas[1] * X[i][0] + np.random.normal(0, sigma)
        linearRow = [ly, X[i][0]]
        linear.append(linearRow)

        qy = alphas[0] + (alphas[1] * (X[i][0] ** 2)) + np.random.normal(0, sigma)
        quadRow = [qy, X[i][0]]
        quadratic.append(quadRow)
    
    
    return (np.array(linear), np.array(quadratic))

        
    


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph

    X = []
    for i in range(1000):
        X.append([200 * np.random.random() - 100])

    X = np.array(X)

    betas = (1, 2)
    alphas = (2, 3)

    sigmas = []

    for i in range(10):
        sigmas.append(10 ** (i - 4))

    mseLinears = []
    mseQuads = []

    for sigma in sigmas:
        sets = synthetic_datasets(betas, alphas, X, sigma)

        linearSet = sets[0]
        quadSet = sets[1]

        mseLinear = compute_betas(linearSet, [1])[0]
        mseLinears.append(mseLinear)
        mseQuad = compute_betas(quadSet, [1])[0]
        mseQuads.append(mseQuad)
    

    plt.plot(sigmas, mseLinears, label = "linear", marker = "o")
    plt.plot(sigmas, mseQuads, label = "quadratic", marker = "o")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Sigma")
    plt.ylabel("Mean Squared Error")

    ##plt.legend("Hello There")

    plt.savefig('mse.pdf')

if __name__ == '__main__':

    dataset = get_dataset("bodyfat.csv")
    print(gradient_descent(dataset, cols=[2,3], betas=[0,0,0]))
    print(iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4))
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
