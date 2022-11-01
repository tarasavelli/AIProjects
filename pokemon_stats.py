import csv
import numpy as np
import random
import matplotlib.pyplot as plt




def load_data(filepath):

    pokemonList = []

    with open(filepath, newline = '') as csvfile:
        pokemon_reader = csv.DictReader(csvfile)
        i = 0
        for pokemon in pokemon_reader:
            row = dict(pokemon)
            row.pop("Legendary")
            row.pop("Generation")
            row["Total"] = int(row.get("Total"))
            row["HP"] = int(row.get("HP"))
            row["Attack"] = int(row.get("Attack"))
            row["Defense"] = int(row.get("Defense"))
            row["Sp. Atk"] = int(row.get("Sp. Atk"))
            row["Sp. Def"] = int(row.get("Sp. Def"))
            row["Speed"] = int(row.get("Speed"))
            row["#"] = int(row.get("#"))
            
            pokemonList.append(row)
            i += 1


    first20 = pokemonList[0:20]
    
    return first20

            




def calculate_x_y(stats):

    ## x is attack stats, and y is defense stats
    x = stats["Attack"] + stats["Sp. Atk"] + stats["Speed"]
    y = stats["Defense"] + stats["Sp. Def"] + stats["HP"]

    return (x,y)


## distance formula function. 
def distance(p1, p2):

    return (((p2[1] - p1[1]) ** 2) + ((p2[0] - p1[0]) ** 2)) ** .5

## helper method to create the distance matrix
def createDistanceMatrix(dataset, m):
    distanceMatrix = [[0 for x in range(m)] for y in range(m)]

    for i in range(m):
        for j in range(m):
            euclDist = distance(dataset[i], dataset[j])
            distanceMatrix[i][j] = euclDist
    
    return distanceMatrix

## cleans the Z matrix to get rid of the last column as well as the first 20 rows
def cleanup(Z, m):
    for i in range(m):
        Z.pop(0)


    for i in range(m - 1):
        Z[i].pop(4) 
    
    return Z
## helper function to get rid of tuples with nan or inf values in them
def clean(dataset):

    tuplesToRemove = []

    for tuple in dataset:
        if(np.isnan(tuple[0]) | np.isnan(tuple[1]) | np.isinf(tuple[0]) | np.isinf(tuple[1])):
            tuplesToRemove.append(tuple)
    
    for tuple in tuplesToRemove:
        dataset.remove(tuple)

    return dataset


def hac(dataset):

    dataset = clean(dataset)

    m = len(dataset)

    ## Creating distance matrix 

    distanceMatrix = createDistanceMatrix(dataset, m)

    ## creating a dictionary that associates a number with each cluster

    Z = []
    clusterList = []

    for i in range(m):
        zRow = []
        zRow.append(i)
        zRow.append(i)
        zRow.append(0)
        zRow.append(1)
        pointList = []
        pointList.append(i)
        zRow.append(pointList)
        Z.append(zRow)
        clusterList.append(i)
    

    currClusterNum = m

    ## loop will pass through data m - 1 times
    for i in range(m - 1):
        
        minDistance = -1
        minc1 = -1
        minc2 = -1

        ## loop compares all clusters that are in the cluster list
        for c1 in clusterList:
            for c2 in clusterList:
                if(c1 != c2):
                    c1points = Z[c1][4]
                    c2points = Z[c2][4]
                    distanceList = []

                    ## getting the minimum distance between points of two clusters
                    for point1 in c1points:
                        for point2 in c2points:
                            distanceList.append(distanceMatrix[point1][point2])
                    
                    if((minDistance == -1) | (min(distanceList) < minDistance)):
                        minDistance = min(distanceList)
                        minc1 = c1
                        minc2 = c2
        ## when the minimum distance and two clusters have been found, remove the two clusters and add the newer one
        clusterList.remove(minc1)
        clusterList.remove(minc2)
        clusterList.append(currClusterNum)

        currClusterNum += 1

        ## adding pass to Z matrix
        newZRow = [minc1, minc2, minDistance, Z[minc1][3] + Z[minc2][3], Z[minc1][4] + Z[minc2][4]]
        Z.append(newZRow)

    ## getting rid of extraneous variables used for computation
    Z = cleanup(Z, m)
    
    return np.array(Z)

        
    


def random_x_y(m):

    randomPokemons = []

    for i in range(m):
        x = random.randint(0, 360)
        y = random.randint(0, 360)

        pokemon = (x,y)
        randomPokemons.append(pokemon)

    return randomPokemons


def imshow_hac(dataset):

    dataset = clean(dataset)

    m = len(dataset)
    xVals = []
    yVals = []
    for tuple in dataset:
        xVals.append(tuple[0])
        yVals.append(tuple[1])
    
    plt.scatter(xVals, yVals)



    ## Creating distance matrix 

    distanceMatrix = createDistanceMatrix(dataset, m)

    ## creating a dictionary that associates a number with each cluster

    Z = []
    clusterList = []

    for i in range(m):
        zRow = []
        zRow.append(i)
        zRow.append(i)
        zRow.append(0)
        zRow.append(1)
        pointList = []
        pointList.append(i)
        zRow.append(pointList)
        Z.append(zRow)
        clusterList.append(i)
    

    currClusterNum = m

    ## loop will pass through data m - 1 times
    for i in range(m - 1):
        
        minDistance = -1
        minc1 = -1
        minc2 = -1

        minPoints = []

        ## loop compares all clusters that are in the cluster list
        for c1 in clusterList:
            for c2 in clusterList:
                if(c1 != c2):
                    c1points = Z[c1][4]
                    c2points = Z[c2][4]
                    distanceList = []

                    distanceDict = {}


                    ## getting the minimum distance between points of two clusters
                    for point1 in c1points:
                        for point2 in c2points:
                            distance = distanceMatrix[point1][point2]
                            distanceList.append(distance)
                            pointList = [dataset[point1], dataset[point2]]
                            if not (distance in distanceDict.keys()):
                                distanceDict.update({distance : pointList})
                    
                    if((minDistance == -1) | (min(distanceList) < minDistance)):
                        minDistance = min(distanceList)
                        minc1 = c1
                        minc2 = c2
                        minPoints = distanceDict.get(minDistance)
                        

        ## when the minimum distance and two clusters have been found, remove the two clusters and add the newer one
        clusterList.remove(minc1)
        clusterList.remove(minc2)
        clusterList.append(currClusterNum)

        currClusterNum += 1

        ## adding pass to Z matrix
        newZRow = [minc1, minc2, minDistance, Z[minc1][3] + Z[minc2][3], Z[minc1][4] + Z[minc2][4]]
        Z.append(newZRow)

        xVals = []
        yVals = []
        for point in minPoints:
            xVals.append(point[0])
            yVals.append(point[1])
        
        
        
        plt.plot(xVals, yVals)
        plt.pause(0.1)
        

    ## getting rid of extraneous variables used for computation
    Z = cleanup(Z, m)
    
    Znump = np.array(Z)

    plt.show()

    




    

    
    
    

