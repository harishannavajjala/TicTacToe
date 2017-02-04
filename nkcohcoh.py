"""

IMP NOTE:- My output is in multiple lines. Please consider the final line as my next move.
(1) I have formulated this problem as an adverserial search problem.
    The state space is the set of all states what are reachable by making possible moves from the given state.
    Successor function:- For any given state our function generates all the possible next moves by placing our stone(white or black) in all the
                        available empty locations. Then we exclude those states which are symmetrical(for example: [[1,0,0],[0,0,0],[0,0,0]] and
                        [[0,0,1],[0,0,0],[0,0,0]] are symmetrical and would ultimately lead to same output. So we have decided to consider only
                        one state from a set of symmetrical states)
    Edge weights: Edge weights are constant. Winning is the primary objective here, but the path doesnt matter. Path couldn't be guaranteed
                    because our adversary would always change the path at some level in our game tree.
    Heuristic: We have desinged an evaluation function which returns either a +value(if max is likely to win) or a -ve value(if min is likely to win)
                or a 0 if it is going to be a draw match. Evaluation value = (#of k contigous blocks available for
                min in all the rows,columns and all diagonals) -  (#of k contigous blocks available for
                max in all the rows,columns and all diagonals)
                 This is opposite of the heuristic that we generally have for tic tac toe.

(2) Search Algorithm:-
    I have used minimax and alpha beta pruning
    Step 1: Generate a set of successors and refine them by using the symmetry factor
    Step 2: Generate a game tree by starting with the input board as a head node and its successors as the head node's children and so on.
             The tree is built in a Depth First manner => If we need to prune under a node N in our implementation we don't even create those nodes
             that need to be pruned.
    Step 3: Stop building the tree if i) If a terminal state is met or ii)The depth exceeds the maximum depth set
    Step 4: Once a leaf node is reached compute the evaluation value for that state and propagate the value up.
    Step 5: Once the whole tree is built from head node(i.e., the initial board) we choose the successor with highest beta value

(3) Problems faced(in the order we faced):-
    1. What kind of data structure should be used to build the game tree (eg:- should we use a class and build actual tree or use dictionaries and lists)
    2. Should the tree be built in a Breadth first manner or a depth first manner
    3. Should we create the nodes which need to be pruned and then prune them while searching ? Or should we not create them in the first place ?
    4. If at head node if we have n successors with same beta value which successor should we choose as out next move?
        our answer:-
                   i) If the beta values are negative then that means max would loose. So take that path which has more depth so that if min makes a
                   bad move max might still win
                   ii) If the beta values are +ve then that means max is probably going to win. So take the path which has sortest depth so that max
                        wins at the earliest

"""
import time
import copy
import numpy
import sys
class Board:								# class to represent each node in the tree
    def __init__(self,board= [],type="max",depthLevel=1,evalue = float("-inf")):
        self.type = type #Default value
        self.board = board
        self.depthLevel =depthLevel
        self.maxDepthBelow=0
        self.parent = None
        self.children = []
        self.evalue = evalue
        #For a max node set evalue default value to -infinity and for min node set it to +infinity

#Given a board this method rotates a given board 90 degrees, 180 degrees and 270 degress and returns the corresponding output states
def rotate_matrix(input_matrix):
    #to rotate
    exec"rotated=zip(*input_matrix[::-1]);"*3
    #to make it list
    for i in range(0, len(rotated)):
        rotated[i] = list(rotated[i])
    return rotated

# Finds the rotated states
def getClones(succ):
    cloneList = []
    s2=rotate_matrix(succ)
    s3 = rotate_matrix(s2)
    s4 = rotate_matrix(s3)

    cloneList.append(s2)
    cloneList.append(s3)
    cloneList.append(s4)

    return cloneList

#This method is to check for symmetry between states
def refineSucc(successors):
    newSuccs = []
    newSuccs.append(successors[0])


    for succ in successors[1:]:
        flag = False
        cloneList=getClones(succ)
        for clone in cloneList:
            if(clone in newSuccs):
                flag = True
                break
        if(not flag):
            newSuccs.append(succ)

    return newSuccs



#Converts a board given as a string into a 2-D list
def getBoard(string,n):
    board=[[0] * n for _ in range(n)]
    i=0
    j=0
    valid = True
    for char in string:
        if char == "w":
            board[i][j]=2
        elif char =="b":
            board[i][j] = 1
        j+=1
        if (j == n):
            i+=1
            j=0
    mx=0
    mn=0
    if(string.count("w") == string.count("b")):
        mx = 2
        mn = 1
    elif(abs(string.count("w") - string.count("b")) > 1):
        valid = False
    else:
        mx = 1
        mn = 2

    return board,mx,mn,valid

#Prints a 2-D list in machine readable format
def printBoard(board):
    string =""
    for i in range(len(board)):
        for j in range(len(board)):
            if(board[i][j]==1):
                string+="b"
            elif(board[i][j] == 2):
                string += "w"
            else:
                string += "."
    return string

#Finds evaluation value for a given board
def evaluationValue(board):
    # white is 2
    global k
    global mx
    global mn
    w_count = 0
    b_count = 0

    #Below for loop gets the row count
    for i in range(len(board)):
        for j in range(len(board)):
            if(j+k <= len(board)):
                s= board[i][j:j+k]
                if(2 not in s):
                    b_count+=1
                if(1 not in s):
                    w_count+=1
            else:
                break

    for j in range(len(board)):
        for i in range(len(board)):
            s=""
            if(i+k <= len(board)):
                for l in range(i,i+k):
                    s+=str(board[l][j])
                if ("2" not in s):
                    b_count += 1
                if ("1" not in s):
                    w_count += 1
            else:
                break


    a = numpy.array(board)
    dags = []
    for i in range(-a.shape[0] + 1, a.shape[1]):
        dags.append(a[::-1, :].diagonal(i))

    for i in range(a.shape[1] - 1, -a.shape[0], -1):
        dags.extend([a.diagonal(i)])

    dags = [n.tolist() for n in dags]

    for diag in dags:
        if (len(diag) >= k):
            for j in range(len(diag)):
                if (j + k <= len(diag)):
                    s = diag[j:j + k]
                    if (2 not in s):
                        b_count += 1
                    if (1 not in s):
                        w_count += 1
                else:
                    break

    if(mx == 2):
        evalue = b_count - w_count
    else:
        evalue = w_count - b_count

    return evalue

#This method checks if the tree below a given input node needs to be pruned(in our case should it be created or not)
def checkAlphaBeta(node):
    evalue = node.evalue
    type = node.type
    while(node.parent !=None):
        if(type=="max" and node.parent.type=="min" and evalue>node.parent.evalue):
            return True #Means prune it
        elif(type=="min" and node.parent.type=="max" and evalue<node.parent.evalue):
            return True  # Means prune it
        node = node.parent
    return False #This is for safety. Code should never actually reach this line.

#Finds successors of a given state
def findSucc(node):
    global mx
    global mn
    board = node.board
    succList = []
    for i in range(len(board)):
        for j in range(len(board)):
            if(board[i][j]==0):
                newBoard = copy.deepcopy(board)
                if(node.type == "max"):
                    newBoard[i][j]=mx
                elif(node.type == "min"):
                    newBoard[i][j] = mn
                succList.append(newBoard)
    return succList

#Finds if a given state is a terminal state or not
def isTerminal(node):
    # white is 2
    global k
    global mx
    global mn

    board = node.board

    # Below for loop gets the row count
    for i in range(len(board)):
        for j in range(len(board)):
            if (j + k <= len(board)):
                s = board[i][j:j + k]
                if(s.count(2)==k or s.count(1)==k):
                    return True
            else:
                break

    for j in range(len(board)):
        for i in range(len(board)):
            s = ""
            if (i + k <= len(board)):
                for l in range(i, i + k):
                    s += str(board[l][j])
                if (s.count('2') == k or (s.count('1') == k and len(s)==k)):
                    return True
            else:
                break


    a = numpy.array(board)
    dags = []
    for i in range(-a.shape[0] + 1, a.shape[1]):
        dags.append(a[::-1, :].diagonal(i))

    for i in range(a.shape[1] - 1, -a.shape[0], -1):
        dags.extend([a.diagonal(i)])

    dags = [n.tolist() for n in dags]

    for diag in dags:
        if(len(diag)>=k):
            for j in range(len(diag)):
                if (j + k <= len(diag)):
                    s = diag[j:j+k]
                    if (s.count(2) == k or s.count(1) == k):
                        return True
                else:
                    break



    # for d in range(len(board)):
    #     s = ""
    #     if (d + k <= len(board)):
    #         for l in range(d, d + k):
    #             s += str(board[l][l])
    #         if (s.count('2') == k or (s.count('1') == k and len(s)==k)):
    #             return True
    #     else:
    #         break
    #
    # i = 0
    # j = len(board) - 1
    # while (0 <= i < len(board) and 0 <= j < len(board)):
    #     ii = i
    #     jj = j
    #     s = ""
    #     if (ii + k <= len(board) and jj - k >= -1):
    #         for n in range(k):
    #             s += str(board[ii][jj])
    #             ii += 1
    #             jj -= 1
    #         if (s.count('2') == k or (s.count('1') == k and len(s) == k)):
    #             return True
    #     else:
    #         break
    #     i+=1
    #     j-=1

    return False

#This is the actual method which recursively builds the game tree
def buildTree(node):
    if(isTerminal(node)):
        if (node.type == "min"):
            node.evalue=float("-inf")
            return node.evalue,node.depthLevel
        elif (node.type == "max"):
            node.evalue = float("inf")
            return node.evalue,node.depthLevel
    if(node.depthLevel > totalDepth):

        evalue = evaluationValue(node.board)
        node.evalue = evalue
        return node.evalue,node.depthLevel
    successors = findSucc(node) # Here successors are boards, not nodes

    if(len(successors)>9):
        successors = refineSucc(successors)

    for board in successors:
        child = Board(board,"min",node.depthLevel+1,float("inf")) if node.type=="max" else Board(board,"max",node.depthLevel+1)
        child.parent = node
        node.children.append(child)

        evalue,maxDepthBelow = buildTree(child)
        if(maxDepthBelow > node.maxDepthBelow):
            node.maxDepthBelow = maxDepthBelow

        if (node.type == "min" and evalue < node.evalue):
            node.evalue = evalue
        elif (node.type == "max" and evalue > node.evalue):
            node.evalue = evalue

        if(checkAlphaBeta(node)):
            return node.evalue,node.maxDepthBelow
    return node.evalue,node.maxDepthBelow

#This method  generates sucessors and finds the best one from them
def findBestSucc(randomSuccList):
    bestSucc = randomSuccList[0]
    bestEval = evaluationValue(bestSucc)
    for succ in randomSuccList[1:]:
        currEval = evaluationValue(succ)
        if(currEval > bestEval):
            bestEval = currEval
            bestSucc = succ

    return bestSucc

#Let white be 2 and black be +1
if __name__ == '__main__':
    totalDepth = 2
    time1 = time.time()
    time2 = time.time()
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    board = sys.argv[3]
    t = int(sys.argv[4])

    if(len(board) != n*n):
        print "The given board length is not equal to n*n. Please re-run the program with a proper input"
        exit()
    board2 = board
    board, mx, mn, valid = getBoard(board,n)
    if mx==2:
        w = "w"
    else:
        w = "b"

    #generates a random successor
    board2=list(board2)
    board2[board2.index(".")] = w
    print("".join(board2))

    #below code finds a best successor using A start approach
    head = Board(board,"max",1)
    randomSuccList = findSucc(head)
    print printBoard(randomSuccList[0])
    print printBoard(findBestSucc(randomSuccList))


    if(isTerminal(head)):
        print "Invalid Input: This is already a terminal state."
        exit()
    print "Last output is the best output"
    while (time2-time1)<(t-0.5):
        buildTree(head)
        bestChild = head.children[0]
        bestValue = bestChild.evalue
        bestDepth = bestChild.maxDepthBelow
        for child in head.children[1:]:
            if(child.evalue > bestValue):
                bestChild = child
                bestValue = child.evalue
                bestDepth = child.maxDepthBelow
            elif(child.evalue == bestValue):
                if(bestValue <0):
                    if(child.maxDepthBelow>bestDepth):
                        bestChild = child
                        bestValue = child.evalue
                        bestDepth = child.maxDepthBelow
                elif(bestValue > 0):
                    if (child.maxDepthBelow < bestDepth):
                        bestChild = child
                        bestValue = child.evalue
                        bestDepth = child.maxDepthBelow


        if(isTerminal(bestChild)):
            print "GAME OVER: Max Lost"
        print(printBoard(bestChild.board))
        totalDepth+=2
        time2 = time.time()

    print "BELOW IS THE FINAL OUTPUT:"
    print(printBoard(bestChild.board))
        ## By this line tree is built and head is the given input state


