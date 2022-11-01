import random
import copy
import math

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
    
    def check_state(self,state):

        ## checking if there ar eight pieces on the board. If not, then we are in the drop phasee, if there is, we are past drop phase
        sum = 0
        for i in range(5):
            for j in range(5):
                if((state[i][j] == 'r') | (state[i][j] == 'b')):
                    sum += 1
        
        return sum < 8
    
    ## method returns a tuple containing the original coordinates and the moves that can be made with the piece in those coordinates
    def tuple_list(self, piece_coords, state):
        possMoves = []
        x = piece_coords[0]
        y = piece_coords[1]

        possMoves.append((x + 1, y))
        possMoves.append((x - 1, y))
        possMoves.append((x, y + 1))
        possMoves.append((x, y - 1))
        possMoves.append((x + 1, y + 1))
        possMoves.append((x - 1, y + 1))
        possMoves.append((x + 1, y - 1))
        possMoves.append((x - 1, y - 1))

        moves = []
        for move in possMoves:
            if ((move[0] >= 0) and (move[0] < 5) and (move[1] >= 0) and (move[1] < 5)):
                if(state[move[0]][move[1]] == ' '):
                    moves.append(move)


        return (piece_coords, moves)


    
    def gen_succ(self, state, drop):


        successors = []
        if drop:
            move = copy.deepcopy(state)
            for i in range(5):
                for j in range(5):
                    if move[i][j] == ' ':
                        successors.append(((-1, -1), (i, j)))
        else:
            move = copy.deepcopy(state)
            for i in range(5):
                for j in range(5):
                    ## testing to see if the the piece is my piece (black or red)
                    if(move[i][j] == self.my_piece):
                        coords = (i, j)
                        ## coming up with a list of possible moves from that position
                        actions = self.tuple_list(coords, state)[1]
                        ## carrying out the moves and adding them to the successors list. 
                        for action in actions:
                            ##successors.append(move)
                            ## each successorr will noow be a tuple pair with the old and new coordinates of a piece
                            successors.append((coords, action))
        
        return successors

    ## will create a state based on a move (will move the piece from the old coordinates to new if in continuing phase or place piece in new coordinates if in drop phase)
    def createState(self, state, prev_coords, new_coords, drop):
        stateCopy = copy.deepcopy(state)

        if not drop:
            stateCopy[prev_coords[0]][prev_coords[1]] = ' '
            stateCopy[new_coords[0]][new_coords[1]] = self.my_piece
        else:
            stateCopy[new_coords[0]][new_coords[1]] = self.my_piece

        return stateCopy
    
    def floorDist(self, p1, p2):
        return math.floor(math.sqrt( ((p1[0]-p2[0])**2) + ((p1[1]-p2[1]) ** 2) ))

    ## evaluates how close to winning one is by essentially finding the first piece that aligns with my_piece, and then assigning weights .5, .25, .125, ... 
        # to pieces that are 1, 2, 3, ... pieces away. These values are then added and divided by 4 if the color is my piece or by -4 if the color is the other side. Will create a negative result for the other side and will return the value that is greater absolutely
    def hueristic_game_value(self, state):

        ai_coord = ()
        opp_coord = ()
        ai_hueristic = 0.0
        opp_hueristic = 0.0
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    ai_coord = (i, j)

                if (state[i][j] != self.my_piece) and (state[i][j] != ' '):
                    opp_coord = (i, j)
            
            if((len(ai_coord) != 0) and (len(opp_coord)!= 0)):
                break

        
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    distance = self.floorDist(ai_coord, (i, j))
                    if distance != 0:
                        ai_hueristic += (1 / (2 ** distance))
                elif state[i][j] != ' ':
                    distance = self.floorDist(opp_coord, (i, j))
                    if distance != 0:
                        opp_hueristic += (1 / (2 ** distance))
            

        if ai_hueristic > opp_hueristic:
            return (ai_hueristic + 1) / 4
        else:
            return (opp_hueristic + 1) / - 4

    ## returns a value associated with the move as well as the move to make in (old_coords, new_coords) form as a tuple. 
    def max_value(self, state, depth, alpha, beta):
        
        if self.game_value(state) != 0:
            return (self.game_value(state), ())
        
        if depth == 0:
            return (self.hueristic_game_value(state), ())
        
        maxEval = -math.inf

        ## handles drop check internally
        drop = self.check_state(state)
        bestMove = ()
        for successor in self.gen_succ(state, drop):
            testState = self.createState(state, successor[0], successor[1], drop)
            eval = self.min_value(testState, depth - 1, alpha, beta)[0]
            if eval > maxEval:
                bestMove = successor
                maxEval = eval

            alpha = max(alpha, eval)
            if beta <= alpha:
               break

        return (maxEval, bestMove)

    ## similar to above
    def min_value(self, state, depth, alpha, beta):
        if self.game_value(state) != 0:
            return (self.game_value(state), ())
        
        if depth == 0:
            return (self.hueristic_game_value(state), ())
        
        minEval = math.inf

        drop = self.check_state(state)
        bestMove = ()
        for successor in self.gen_succ(state, drop):
            testState = self.createState(state, successor[0], successor[1], drop)
            eval = self.max_value(testState, depth - 1, alpha, beta)[0]
            if eval < minEval:
                bestMove = successor
                minEval = eval

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return (minEval, bestMove)




        



    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        move = self.max_value(state, 2, -math.inf, math.inf)

        ## drop check makes it such that only the new_coords are returned (as there are no source coordinates)
        if(self.check_state(state)):
            return [move[1][1]]
        

        return [move[1][1], move[1][0]]

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and diamond wins
        """
        ## diagonal check
        for i in range(2):
            for j in range(2):
                if (state[i][j] != ' ') and (state[i][j] == state[i + 1][j + 1] == state[i + 2][j + 2] == state[i + 3][j + 3]):
                    return 1 if state[i][j] == self.my_piece else -1
                if(state[i][4 - j] != ' ' and state[i][4 - j] == state[i + 1][3 - j] == state[i + 2][2 - j] == state[i + 3][1 - j]):
                    return 1 if state[4 - i][4-j] == self.my_piece else -1
        
        ## diamond check
        for i in range(3):
            for j in range(3):
                if (state[i][j + 1] != ' ') and (state[i][j + 1] == state[i + 1][j] == state[i + 2][j + 1] == state[i + 1][j + 2]):
                    return 1 if state[i][j + 1] == self.my_piece else -1


        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        # TODO: check / diagonal wins
        # TODO: check diamond wins

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()

