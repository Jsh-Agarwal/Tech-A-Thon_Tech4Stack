import cv2
import numpy as np
import heapq

def preprocess_maze(image_path):
    """Preprocess the maze image to create a binary representation."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    _, binary = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
    return binary

def get_neighbors(node, maze):
    """Get valid neighbors for Dijkstra's algorithm."""
    rows, cols = maze.shape
    x, y = node
    neighbors = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 1:
            neighbors.append((nx, ny))
    return neighbors

def dijkstra(maze, start, end):
    """Solve the maze using Dijkstra's algorithm."""
    rows, cols = maze.shape
    dist = {start: 0}
    prev = {start: None}
    visited = set()
    pq = [(0, start)]  # Priority queue: (distance, node)

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)

        # If we reach the end, reconstruct the path
        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = prev[current_node]
            return path[::-1]  # Reverse the path

        # Explore neighbors
        for neighbor in get_neighbors(current_node, maze):
            if neighbor in visited:
                continue
            new_dist = current_dist + 1  # All edges have weight 1
            if neighbor not in dist or new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    return []  # Return an empty list if no path is found

def draw_path(maze_image, path):
    """Draw the solution path on the maze image."""
    for x, y in path:
        maze_image[x, y] = [0, 0, 255]  # Mark path in red
    return maze_image

def solve_maze(image_path, start, end):
    """Solve the maze and output the solution."""
    # Preprocess the maze
    binary_maze = preprocess_maze(image_path)
    print(f"Binary maze shape: {binary_maze.shape}")

    # Solve the maze using Dijkstra's algorithm
    path = dijkstra(binary_maze, start, end)
    if not path:
        print("No solution found.")
        return

    print(f"Path found with {len(path)} steps.")
    
    # Draw the path on the original maze image
    original_image = cv2.imread(image_path)
    for x, y in path:
        original_image[x, y] = [0, 0, 255]  # Red for the solution path

    # Display and save the solution
    cv2.imshow("Solved Maze", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("solved_maze.jpg", original_image)

# Define the start and end points
image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Tech-A-Thon_Tech4Stack\testing data\maze.jpg"
start = (1, 1)  # Example start point, adjust as needed
end = (20, 20)  # Example end point, adjust as needed

# Solve the maze
solve_maze(image_path, start, end)