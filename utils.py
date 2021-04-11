import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


### Heuristic functions ###
def manhattan_distance(state_a, state_b):
    """
    Args:
        state_a: tuple (x1, y1)
        state_b: tuple (x2, y2)
    Returns:
        L1 distance
    """
    return np.sum(np.abs(np.array(state_a) - np.array(state_b)))


### Visualization ###
def show(maze):
    """Generate a simple image of the maze."""
    grid = maze.grid.copy()
    if maze.start is not None:
        grid[maze.start] = 2
    if maze.end is not None:
        grid[maze.end] = 3
    if maze.solutions is not None:
        for cell in maze.solutions[0]:
            grid[cell] = 4
    
    plt.figure(figsize=(10, 5))
    cmap = colors.ListedColormap(['white', 'black', 'blue', 'red', 'yellow'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(grid, cmap=cmap, interpolation='nearest', norm=norm)
    plt.show()