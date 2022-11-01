from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

n = 2414 ## number of images
d = 1024 ## number of features per image


def load_and_center_dataset(filename):
    # TODO: add your code here
    faceData = np.load(filename)

    ## then must center the data (subtract the mean of each column from each element in the column)

    centeredFaceData = faceData - np.mean(faceData, axis = 0)

    return centeredFaceData



def get_covariance(dataset):
    # TODO: add your code here

    datasetTranspose = np.transpose(dataset)

    covarianceMatrix = np.dot(datasetTranspose, dataset) / (n - 1)

    return covarianceMatrix

    




def get_eig(S, m):
    # TODO: add your code here
    eigenvaluesAndVectors = eigh(S, subset_by_index = [d - m, d - 1])
    eigenVals = eigenvaluesAndVectors[0]
    eigenVectors = eigenvaluesAndVectors[1]

    ## creating diagonal matrix of eigen vals 
    diagonalMatrix = np.diag(np.flip(eigenVals))
    
    correctedEigenVectors = []

    ## flipping the rows of the eigen vectors in order to have the correct correspondance
    for i in range(len(eigenVectors)):
        row = eigenVectors[i]
        reversedRow = np.flip(row)
        correctedEigenVectors.append(reversedRow)

    
    
    correctedEigenVectors = np.array(correctedEigenVectors)
       

    return diagonalMatrix, correctedEigenVectors

def get_eig_perc(S, perc):

    eigenvalsandVects = eigh(S)
    sumEigenVals = sum(eigenvalsandVects[0])

    ## getting the number of eigenvals that meet the percentage threshold
    numEigenVals = 0

    for eigenval in eigenvalsandVects[0]:
        if(eigenval / sumEigenVals > perc):
            numEigenVals += 1
    

    percValsAndVects = eigh(S, subset_by_index = [d - numEigenVals, d - 1])
    
    eigenVals = np.flip(percValsAndVects[0])

    ## creating eigenmatrix of eigenvalues that meet the percent threshold
    eigenVectors = percValsAndVects[1]

    eigenMatrix = np.diag(eigenVals)


    eigenVectMatrix = []

    for i in range(len(eigenVectors)):
        row = eigenVectors[i]
        reversedRow = np.flip(row)
        eigenVectMatrix.append(reversedRow)

    return np.array(eigenMatrix), np.array(eigenVectMatrix)
            






def project_image(img, U):

    Utrans = np.transpose(U)

    alphas = []

    for i in range(len(Utrans)):
        alpha = np.dot(Utrans[i], img)
        alphas.append(alpha)


    projectedImage = [0] * d

    for i in range(len(Utrans)):
        singleProjectedComponent = alphas[i] * Utrans[i]
        projectedImage += singleProjectedComponent
    
    return np.array(projectedImage)


def display_image(orig, proj):
    reshapedOrig = np.reshape(orig, (32, 32))
    reshapedProj = np.reshape(proj, (32, 32))

    figure, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Original")
    ax2.set_title("Projection")

    originalImage = ax1.imshow(np.transpose(reshapedOrig), aspect = 'equal')
    projectedImage = ax2.imshow(np.transpose(reshapedProj), aspect = 'equal')

    figure.colorbar(originalImage, ax = ax1)
    figure.colorbar(projectedImage, ax = ax2)

    plt.show()
