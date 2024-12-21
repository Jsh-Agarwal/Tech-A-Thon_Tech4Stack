import collections

def bfs_maze_solver(maze, start, end):
    """
    Solve a maze using Breadth-First Search (BFS).
    
    :param maze: 2D NumPy array of the maze (1: path, 0: wall)
    :param start: Tuple (x, y) for the starting point
    :param end: Tuple (x, y) for the ending point
    :return: The maze with the path marked
    """
    rows, cols = maze.shape
    queue = collections.deque([start])  # Initialize the queue with the starting position
    visited = set()
    visited.add(start)
    parent = {}  # To reconstruct the path

    # BFS loop
    while queue:
        current = queue.popleft()

        if current == end:
            break

        for neighbor in get_neighbors_bfs(current, maze, visited):
            queue.append(neighbor)
            visited.add(neighbor)
            parent[neighbor] = current

    # Reconstruct the path from end to start
    path = []
    current = end
    while current in parent:
        path.append(current)
        current = parent[current]
    path.append(start)
    path.reverse()

    # Mark the path on the maze
    for x, y in path:
        maze[x, y] = 2  # Mark the path with 2

    return maze, path

def get_neighbors_bfs(node, maze, visited):
    """Get valid neighbors for BFS."""
    rows, cols = maze.shape
    x, y = node
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 1 and (nx, ny) not in visited:
            neighbors.append((nx, ny))
    return neighbors

# Example usage
if __name__ == "__main__":
    import cv2
    import numpy as np

    # Preprocess the maze
    def preprocess_maze(image_path):
        """Convert maze image to binary matrix."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        _, binary = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
        return binary

    image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Tech-A-Thon_Tech4Stack\testing data\maze.jpg"
    binary_maze = preprocess_maze(image_path)

    # Define start and end points
    start = (1, 1)  # Update this as per your maze
    end = (binary_maze.shape[0] - 2, binary_maze.shape[1] - 2)  # Update this as per your maze

    # Solve the maze
    solved_maze, path = bfs_maze_solver(binary_maze, start, end)

    if path:
        print(f"Path found with {len(path)} steps.")
        # Convert to visualized output
        solved_image = cv2.imread(image_path)
        for x, y in path:
            solved_image[x, y] = [0, 0, 255]  # Mark path in red
        cv2.imshow("Solved Maze", solved_image)
        cv2.imwrite("solved_maze_bfs.jpg", solved_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No solution found.")