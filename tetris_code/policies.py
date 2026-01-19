import copy 
import numpy as np

DEBUG = False

def heights(board):
    """Compute the "height" of each colum, which is the number of consecutive zeros on this column starting from the top of the board

    Args:
        board: the state of the board 

    Returns: heights_column (list of int): the height of each column 

    """
    heights_column = [] 
    #loop over the columns of the board
    for i in range(board.shape[1]):
        #ignore the columns that form the "padding" of the board, they have a "1" on top
        if (board[0,i] != 1):
            #start by the second highest point of the current column
            j = 2
            #go down along the column until a non "0" pixel is encountered
            while (board[j,i] == 0) and (j < board.shape[0]): j = j+1
            #store the result
            heights_column.append(j) 
    return(heights_column) 

def holes(board):
    """ Compute the number of holes on the board, which is the number of pixels containing "0" and such that the pixel directly above them does not contain a "0$

    Args:
        board: the state of the board 

    Returns: nb_holes (int): the number of holes on the board 

    """
    nb_holes = 0
    #loop over the lines and columns of the board
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            #if board[i,j] = 0 and board[i-1,j] != 0 then pixel [i,j] is a hole
            if (i > 1) and (board[i,j] == 0) and (board[i-1,j] != 0): nb_holes = nb_holes + 1
    return nb_holes 

def policy_down(env):
    """ Naive policy always selecting the hard_drop action 

    Args:
        env: the game environment

    Returns: an action (int between 0 and 7)

    """
    return 2

def policy_random(env):
    """ Random policy selecting actions uniformly at random 

    Args:
        env: the game environment

    Returns: an action (int between 0 and 7)

    """
    return env.action_space.sample()

def policy_greedy(env):
    """ Greedy policy selecting actions in order to minimize a combination of the maximal column height as well as the number of holes

    Args:
        env: the game environment

    Returns: an action (int between 0 and 7)

    """
    #enumerate sequences of actions of the form: rotate the tetromino several times, then move the tetromino to the right (or left) several times, then perform a hard drop
    sequences_of_actions_to_try = []
    for k in range(10): 
        sequences_of_actions_to_try.append( [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [1 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3] + [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3] + [1 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3] + [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3] + [1 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3,3] + [0 for i in range(k-1)] + [5] )
        sequences_of_actions_to_try.append( [3,3,3] + [1 for i in range(k+1)] + [5] )
    nb_sequences = len(sequences_of_actions_to_try)
    #for each of those sequence of actions, evaluate the resulting state and compute its score
    scores = np.zeros(nb_sequences) 
    for i in range(nb_sequences):
        sequence = sequences_of_actions_to_try[i]
        #create a deep copy of the current state of the environment in order to evaluate the impact of a sequence of actions
        new_env = copy.deepcopy(env)
        for action in sequence:
            #perform each action in the given sequence of actions to evaluate the end state
            observation, reward, terminated, truncated, info = new_env.step(action)
            #the score is a combination of the minimal height (when the minimal height is 0 the game is lost) and the number of holes (usually lines that have a hole can become impossible to clear)
            scores[i] = min(heights(observation.get("board")))-100*holes(observation.get("board"))
    #find the sequence of actions maximizing the score
    imax = np.argmax(scores)
    best_sequence = sequences_of_actions_to_try[imax] 
    #recomend the first action in the best sequence of actions  
    return best_sequence[0]
