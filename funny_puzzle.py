import heapq

def h(state):

    manhattan = 0
    for i in range(len(state)):
        if(state[i] != 0):
            correctPosition = ((state[i] - 1) % 3, int((state[i] - 1) / 3))
            currentPosition = ((i % 3), int(i / 3))

            distance = abs(correctPosition[1] - currentPosition[1]) + abs(correctPosition[0] - currentPosition[0])
            manhattan += distance

    
    return manhattan



def print_succ(state):

    emptyIndex = state.index(0)

    emptyX = int(emptyIndex % 3)
    emptyY = int(emptyIndex / 3)

    possibleTiles = [(emptyX + 1, emptyY), (emptyX - 1, emptyY), (emptyX, emptyY + 1), (emptyX, emptyY - 1)]

    possMoves = []
    stateCopy = state
    for possible in possibleTiles:
        

        
        if((possible[0] >= 0) & (possible[0] <= 2) & (possible[1] >= 0) & (possible[1] <= 2)):
            newIndex = possible[1] * 3 + possible[0]

            stateCopy[emptyIndex] = stateCopy[newIndex]
            stateCopy[newIndex] = 0

            l = []
            for num in stateCopy:
                l.append(num)

            possMoves.append(l)

            saveVal = stateCopy[emptyIndex]
            stateCopy[emptyIndex] = 0
            stateCopy[newIndex] = saveVal


        
    
    possMoves = sorted(possMoves)
    open = []
    closed = {}
    heapq.heappush(open, ((h(state) + 0), state, (0, h(state), -1)))

    currState = heapq.heappop(open)
    heapq.heappush(open, ((h(state) + 0), state, (0, h(state), -1)))



    for move in possMoves:
        print(move, end = ' ')
        print("h=" + str(h(move)))
    
            

def get_succ(state):    
    emptyIndex = state.index(0)

    emptyX = int(emptyIndex % 3)
    emptyY = int(emptyIndex / 3)

    possibleTiles = [(emptyX + 1, emptyY), (emptyX - 1, emptyY), (emptyX, emptyY + 1), (emptyX, emptyY - 1)]

    possMoves = []
    stateCopy = state
    for possible in possibleTiles:
        

        
        if((possible[0] >= 0) & (possible[0] <= 2) & (possible[1] >= 0) & (possible[1] <= 2)):
            newIndex = possible[1] * 3 + possible[0]

            stateCopy[emptyIndex] = stateCopy[newIndex]
            stateCopy[newIndex] = 0

            l = []
            for num in stateCopy:
                l.append(num)

            possMoves.append(l)

            saveVal = stateCopy[emptyIndex]
            stateCopy[emptyIndex] = 0
            stateCopy[newIndex] = saveVal


        
    
    possMoves = sorted(possMoves)

    return possMoves




def contains(successor, open, closed):

    returnVal = False
    for state in open:
        if state[1] == successor:
            returnVal = True

    for state in closed:
        if state[1] == successor:
            returnVal = True

    return returnVal

def solve(state):
    ## cost = level of successor in heap (1 move away, cost = 1, 2 moves away, cost = 2)
    ## heuristic = manhattan distance

    open = []

    closed = []

    maxQlen = 0

    heapq.heappush(open, (h(state) + 0, state, (0, h(state), -1)))

    path = []
    while len(open) > 0:

        currState = heapq.heappop(open)
        closed.append(currState)

        

        if(h(currState[1]) == 0):
            parentIndex = currState[2][2]
            path.append(currState)

            while parentIndex != -1:
                parent = closed[parentIndex]
                path.append(closed[parentIndex])
                parentIndex = parent[2][2]
        
            break
        
        else:
            successors = get_succ(currState[1])
            
            for successor in successors:
                gn = currState[2][0] + 1
                heuristic = h(successor)

                total = heuristic + gn

                if not contains(successor, closed, open):
                    heapq.heappush(open, (total, successor, (gn, heuristic, len(closed) - 1)))
                    maxQlen += 1



    
    path.reverse()
    for state in path:
        print(state[1], "h={0}".format(state[2][1]), "moves: {0}".format(state[2][0]))
    
    ##print("Max queue length: {0}".format(maxQlen)) (Optional Output)
                
    
                
            
         


if __name__ == "__main__":
    ##print_succ([8,7,6,5,4,3,2,1,0])
    solve([4,3,8,5,1,6,7,2,0])
    ##solve([1,2,3,4,5,6,7,0,8])

