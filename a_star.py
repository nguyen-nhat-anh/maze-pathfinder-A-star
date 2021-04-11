import heapq
from dataclasses import dataclass
from mazelib.solve.MazeSolveAlgo import MazeSolveAlgo


@dataclass
class Node:
    """
    A struct represent a node in search tree
    """
    state: tuple
    g: float = 0.0
    h: float = 0.0
    parent: object = None
        
    def __repr__(self):
        return str(self.state)
    
    def __le__(self, other): # define <= operator
        return (self.g + self.h) <= (other.g + other.h)
    
    def __lt__(self, other): # define < operator
        return (self.g + self.h) < (other.g + other.h)
    
def reconstruct_path(end):
    """
    Args:
        end: goal node
    Returns:
        a list of state from start to goal represents the solution path
    """
    solution = list()
    current = end
    while True:
        solution.append(current.state)
        current = current.parent
        if current is None:
            break
    solution.reverse() # (end -> start) to (start -> end)
    return solution[1:-1] # exclude start and end

class Astar(MazeSolveAlgo):
    def __init__(self, dist, verbose, *args, **kwargs):
        """
        Args:
            dist: heuristic function, accept current state and goal state as inputs and
                  return estimated cost from current to goal
            verbose: boolean, whether to print information each step
        """
        super(Astar, self).__init__(*args, **kwargs)
        self.dist = dist
        self.verbose = verbose
        self.grid = None
        self.start = None
        self.end = None
        
    def find_children(self, node):
        """
        Args:
            node: Node object - node to explore
        Returns:
            children of that node, a list of node state
        """
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        children = []
        for move in moves:
            child_state = tuple(map(sum, zip(node.state, move)))
            # check valid state
            if (not 0 <= child_state[0] < self.grid.shape[0]) or (not 0 <= child_state[1] < self.grid.shape[1]):
                continue # outside maze
            if (child_state == self.end or self.grid[child_state] == 0): # goal or not wall
                if (node.parent is not None) and (child_state == node.parent.state):
                    continue
                children.append(child_state)
        return children
    
    def _solve(self):
        """
        Returns:
            a list of solutions, each solution is a list of state 
            from start to goal represents the solution path
        """
        ################
        # A* algorithm #
        ################
        d = self.dist
        # initialization
        current = Node(self.start, g=0, h=d(self.start, self.end))
        frontier = list() # priority queue, a list of (f value, node)
        heapq.heappush(frontier, (current.g + current.h, current))
        
        explored = list() # set of visited node
        
        if self.verbose:
            step = 0
        while len(frontier) > 0:
            if self.verbose:
                print(f'### Step {step} ###')
                print(f'frontier: {frontier}')
                print(f'explored: {explored}')
            
            # choose the lowest cost node in frontier
            f, current = heapq.heappop(frontier)
            if self.verbose:
                print(f'current: {current}')
                step += 1
            
            # goal test
            if current.state == self.end:
                solution = reconstruct_path(current)
                return [solution]
            
            # mark current node as visited
            explored.append(current)
            
            for child_state in self.find_children(current):
                path_cost = current.g + 1 # all moves cost 1
                
                open_check = next((item for item in frontier if item[1].state == child_state), False)
                if open_check: # check if child node is already in frontier
                    old_path_cost = open_check[1].g
                    if path_cost < old_path_cost: # if found a better path
                        frontier.remove(open_check) # remove from frontier (re-add later with updated path cost)
                        heapq.heapify(frontier)
                        
                closed_check = next((n for n in explored if n.state == child_state), False)
                if closed_check: # check if child node is already in explored
                    old_path_cost = closed_check.g
                    if path_cost < old_path_cost: # if found a better path
                        explored.remove(open_check) # remove from explored (to re-open it later)
                        
                # not in both frontier and explored
                if ((child_state not in [item[1].state for item in frontier]) and 
                    (child_state not in [n.state for n in explored])):
                    # create child node
                    child_node = Node(state=child_state, 
                                      g=path_cost,
                                      h=d(child_state, self.end),
                                      parent=current)
                    # add to frontier
                    heapq.heappush(frontier, (child_node.g + child_node.h, child_node))