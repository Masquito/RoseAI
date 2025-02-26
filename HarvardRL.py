from typing import List
import random

class State:
    def __init__(self, col : int, row : int, character : str):
        self.col = col
        self.row = row
        self.chr = character

class Action:
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    def __init__(self, state : State):
        self.direction = direction

class Agent:
    def __init__(self, state : State):
        self.state : State = state

class Environment:
    startState : State
    endState : State
    states : List[State]
    def __init__(self, file):
        self.states = []
        self.maze = file.read().splitlines()  
        for x, row in enumerate(self.maze):  
            for y, char in enumerate(row):   
                if char == "A":
                    self.startState = State(x, y, "A")
                elif char == "B":
                    self.endState = State(x, y, "B")
                elif char == "#":
                    self.states.append(State(x, y, "#"))
                else:
                    self.states.append(State(x, y, " "))

    def R(self, state : State, action : Action, statePrime : State) -> int:
        if statePrime.chr == "#":
            return 0
        if (self.endState.col - statePrime.col < self.endState.col - state.col) or (self.endState.row + statePrime.row < self.endState.row + state.row):
            return 1
        else:
            return 0

class AI:
    @staticmethod
    def Actions(state : State) -> List[str]:
        actions = [Action.up, Action.down, Action.left, Action.right]
        if state.col == 0:
            actions.remove(Action.left)
        if state.col == len(env.maze[0]) - 1:
            actions.remove(Action.right)
        if state.row == 0:
            actions.remove(Action.up)
        if state.row == len(env.maze) - 1:
            actions.remove(Action.down)
        return actions

if __name__ == "__main__":
    file = open("maze1.txt", "r")
    env = Environment(file)
    file.close()
    agent = Agent(env.startState)
    goal = env.endState
    score = 0
    while(agent.state != goal):
        options = {}
        for x in AI.Actions(agent.state):
            match x:
                case Action.up:
                    stateS = State(agent.state.col, agent.state.row - 1, " ")
                    for x in env.states:
                        if x.col == stateS.col and x.row == stateS.row:
                            stateS = x
                            break;
                    reward = env.R(agent.state, x, stateS)
                case Action.down:
                    stateS = State(agent.state.col, agent.state.row + 1, " ")
                    for x in env.states:
                        if x.col == stateS.col and x.row == stateS.row:
                            stateS = x
                            break;
                    reward = env.R(agent.state, x, stateS)
                case Action.left:
                    stateS = State(agent.state.col - 1, agent.state.row, " ")
                    for x in env.states:
                        if x.col == stateS.col and x.row == stateS.row:
                            stateS = x
                            break;
                    reward = env.R(agent.state, x, stateS)
                case Action.right:
                    stateS = State(agent.state.col + 1, agent.state.row, " ")
                    for x in env.states:
                        if x.col == stateS.col and x.row == stateS.row:
                            stateS = x
                            break;
                    reward = env.R(agent.state, x, stateS)
            options[x] = reward
        agent.state = max(options, key=options.get)
        print(agent.state.row, agent.state.col)

            
        
