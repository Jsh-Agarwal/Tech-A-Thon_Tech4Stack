import math
import heapq

def get_neighbors_astar(node, maze, visited):
    """Get valid neighbors for A* algorithm."""
    rows, cols = maze.shape
    x, y = node
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 1 and (nx, ny) not in visited:
            neighbors.append((nx, ny))
    return neighbors

def astar_maze_solver(maze, start, end):
    """
    Solve a maze using A* algorithm.
    :param maze: 2D NumPy array of the maze (1: path, 0: wall)
    :param start: Tuple (x, y) for the starting point
    :param end: Tuple (x, y) for the ending point
    :return: List of tuples representing the shortest path
    """
    path = []
    visited = set()
    pq = []  # Priority queue: (f_score, g_score, node)

    # Initialize g_score and f_score dictionaries
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    # Push the starting node to the priority queue
    heapq.heappush(pq, (f_score[start], 0, start))
    prev_nodes = {start: None}

    while pq:
        _, current_g, current = heapq.heappop(pq)

        # If we reach the end, reconstruct the path
        if current == end:
            while current:
                path.append(current)
                current = prev_nodes[current]
            return path[::-1]  # Reverse the path

        visited.add(current)

        # Explore neighbors
        for neighbor in get_neighbors_astar(current, maze, visited):
            tentative_g = current_g + 1  # Distance from start to neighbor
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                prev_nodes[neighbor] = current
                heapq.heappush(pq, (f_score[neighbor], tentative_g, neighbor))

    return []  # Return an empty path if no solution is found

def heuristic(node, goal):
    """Heuristic function for A*: Manhattan distance."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def draw_path_on_maze(maze, path):
    """Draw the solution path on the maze."""
    for x, y in path:
        maze[x, y] = 2  # Mark the path with 2
    return maze

# Example usage
if __name__ == "__main__":
    import cv2
    import numpy as np

    # Load and preprocess the maze
    def preprocess_maze(image_path):
        """Preprocess the maze image to create a binary maze."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        _, binary = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
        return binary

    image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Tech-A-Thon_Tech4Stack\testing data\maze.jpg"
    binary_maze = preprocess_maze(image_path)

    # Define start and end points
    start = (1, 1)  # Update as per your maze
    end = (binary_maze.shape[0] - 2, binary_maze.shape[1] - 2)  # Update as per your maze

    # Solve the maze
    solution_path = astar_maze_solver(binary_maze, start, end)

    if solution_path:
        print(f"Path found with {len(solution_path)} steps.")
        # Draw the solution path
        solved_maze = draw_path_on_maze(binary_maze.copy(), solution_path)

        # Visualize and save the solved maze
        solved_image = cv2.imread(image_path)
        for x, y in solution_path:
            solved_image[x, y] = [0, 0, 255]  # Mark the path in red
        cv2.imshow("Solved Maze", solved_image)
        cv2.imwrite("solved_maze.jpg", solved_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No solution found.")