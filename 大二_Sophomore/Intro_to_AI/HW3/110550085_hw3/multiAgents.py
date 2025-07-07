from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        '''
        I implemented this function with the algorithm given in the lecture.
        I start the recurrence by setting the initial depth to 1, state to
        the given gameState, and agentIndex to represent the Pacman.
        When the state is Lose or Win, or when the depth is larger than 
        self.depth, return the current state's evaluationFunction's value.
        If not, then check the agentIndex, if it's 0, then it's time for the
        Pacman, choose the maximum in all the possible choices, if it's not 0, 
        then it's time for one of the ghosts, return the minimum in all 
        possible choices as the optimal choice by the ghost.
        '''

        def minimax(depth, state, agentIndex):
            if(state.isLose() or state.isWin()):
                return self.evaluationFunction(state)
            elif(depth>self.depth):
                return self.evaluationFunction(state)
            
            legal_actions=state.getLegalActions(agentIndex)
            possible_choice=[]
            for action in legal_actions:
                next_state=state.getNextState(agentIndex, action)
                if(agentIndex==state.getNumAgents()-1):
                    possible_choice.append(minimax(depth+1, next_state, 0))
                else:
                    possible_choice.append(minimax(depth, next_state, agentIndex+1))

            #Pacman Action
            if(agentIndex==0):
                if(depth!=1):
                    bestchoice=max(possible_choice)
                    return bestchoice
                else:
                    bestchoice=max(possible_choice)
                    for i in range(len(possible_choice)):
                        if(possible_choice[i]==bestchoice):
                            return legal_actions[i]

            #Ghost Action
            else:
                bestchoice=min(possible_choice)
                return bestchoice

        return minimax(1, gameState, 0)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        '''
        Compared to Part1, I add alpha and beta, and initialize them to -inf and
        inf, respectively. When checking for Pacman, if the current value is larger
        than beta, return the current value(pruning), otherwise, set alpha to the 
        larger one of current value and the current alpha. And for the Ghosts, if
        the current value is smaller than alpha, return the current value(pruning), 
        otherwise, set beta to the smaller one of the current value and the current beta.
        '''
        def ABpruning(depth, state, agentIndex, a, b):
            if(state.isLose() or state.isWin()):
                return self.evaluationFunction(state)
            elif(depth>self.depth):
                return self.evaluationFunction(state)
            
            legal_actions=state.getLegalActions(agentIndex)
            possible_choice=[]
            for action in legal_actions:
                next_state=state.getNextState(agentIndex, action)
                if(agentIndex==state.getNumAgents()-1):
                    x=ABpruning(depth+1, next_state, 0, a, b)
                else:
                    x=ABpruning(depth, next_state, agentIndex+1, a, b)

                if(agentIndex==0):
                    if(x>b):
                        return x
                    else:
                        a=max(a, x)
                else:
                    if(x<a):
                        return x
                    else:
                        b=min(b, x)

                possible_choice.append(x)

            #Pacman Action
            if(agentIndex==0):
                if(depth!=1):
                    bestchoice=max(possible_choice)
                    return bestchoice
                else:
                    bestchoice=max(possible_choice)
                    for i in range(len(possible_choice)):
                        if(possible_choice[i]==bestchoice):
                            return legal_actions[i]
                        
            #Ghost Action
            else:
                bestchoice=min(possible_choice)
                return bestchoice

        return ABpruning(1, gameState, 0, float('-Inf'), float('Inf'))
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        '''
        This part has almost the same concept with part 1, the only difference is
        that for the Ghosts, I choose the expect value instead of the minimum value.
        '''
        def Expectimax(depth, state, agentIndex):
            if(state.isLose() or state.isWin()):
                return self.evaluationFunction(state)
            elif(depth>self.depth):
                return self.evaluationFunction(state)
            
            legal_actions=state.getLegalActions(agentIndex)
            possible_choice=[]

            for action in legal_actions:
                next_state=state.getNextState(agentIndex, action)
                if(agentIndex==state.getNumAgents()-1):
                    possible_choice.append(Expectimax(depth+1, next_state, 0))
                else:
                    possible_choice.append(Expectimax(depth, next_state, agentIndex+1))

            #Pacman Action
            if(agentIndex==0):
                if(depth!=1):
                    bestchoice=max(possible_choice)
                    return bestchoice
                else:
                    bestchoice=max(possible_choice)
                    for i in range(len(possible_choice)):
                        if(possible_choice[i]==bestchoice):
                            return legal_actions[i]

            #Ghost Choose Expect Value
            else:
                expectchoice=float(sum(possible_choice)/len(possible_choice))
                return expectchoice

        return Expectimax(1, gameState, 0)

        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme dipshxt evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    '''
    In this part, I use hope to represent the value to return, and initialize
    it to the current score. Then, I give some weight to the relation to the 
    following parameters:
    1. The minimum Manhattan distance to a ghost.
    2. The remaining time for ghosts to be scared.
    3. The minimum Manhattan distance to a food.
    4. The number of the remaining capsules.
    For each weight, I test many times to get a highest score to achieve.
    And for the number of capsules, I found that no matter how the weight 
    changes, it would yield the same result, but I still keep it there 
    because I think that the tendency to get a capsule is a buff for Pacman
    andmight play a role in the hidden testdata.
    '''
    PacmanPosition=currentGameState.getPacmanPosition()
    Capsules=currentGameState.getCapsules()
    NumFood=currentGameState.getNumFood()
    Food=currentGameState.getFood()
    GhostStates=currentGameState.getGhostStates()

    hope=currentGameState.getScore()

    #Relation with Ghosts
    minDistance=float('Inf')
    for i in GhostStates:
        d=manhattanDistance(PacmanPosition, i.getPosition())
        if(d<minDistance):
            minDistance=d
    RemainingTime=0
    for i in GhostStates:
        RemainingTime+=i.scaredTimer
    if(minDistance!=0):
        hope+=(RemainingTime*10)/minDistance
        if(RemainingTime==0):
            hope-=4/minDistance

    #Relation with Food
    minFoodDistance=float('Inf')
    for i in Food.asList():
        d=manhattanDistance(PacmanPosition, i)
        if(d<minFoodDistance):
            minFoodDistance=d
    
    if(NumFood!=0):
        hope+=5/minFoodDistance

    #Relation with Capsules
    hope-=100*len(Capsules)

    return hope
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
