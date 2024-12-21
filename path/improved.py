import cv2
import numpy as np
from collections import deque

# Load the maze image
maze_image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Tech-A-Thon_Tech4Stack\testing data\maze.jpg"
maze_image = cv2.imread(maze_image_path)

# Convert the image to grayscale for processing
gray_maze = cv2.cvtColor(maze_image, cv2.COLOR_BGR2GRAY)

# Threshold to create a binary image (black and white)
_, binary_maze = cv2.threshold(gray_maze, 128, 255, cv2.THRESH_BINARY_INV)

# Display the dimensions of the maze
print(f"The maze dimensions are: {binary_maze.shape}")

# Extract the red (start) and green (end) points using color thresholds
# Define the color ranges in BGR format for red and green
red_lower = np.array([0, 0, 150])
red_upper = np.array([100, 100, 255])

green_lower = np.array([0, 150, 0])
green_upper = np.array([100, 255, 100])

# Mask for red and green areas
red_mask = cv2.inRange(maze_image, red_lower, red_upper)
green_mask = cv2.inRange(maze_image, green_lower, green_upper)

# Find the coordinates of the red (start) and green (end) points
red_coords = np.argwhere(red_mask == 255)
green_coords = np.argwhere(green_mask == 255)

# Average the coordinates to get the central point of the arrows
start_point = tuple(red_coords.mean(axis=0).astype(int))
end_point = tuple(green_coords.mean(axis=0).astype(int))

print(f"Start point: {start_point}")
print(f"End point: {end_point}")

# Convert the binary maze image to a navigable matrix
# Represent walls as 0 and paths as 1
maze_matrix = (binary_maze // 255).astype(np.uint8)

# Adjust start and end points to be in matrix coordinates (row, col)
start = (start_point[0], start_point[1])
end = (end_point[0], end_point[1])

# Ensure the start and end points are valid (on paths, not walls)
maze_matrix[start[0], start[1]] = 1
maze_matrix[end[0], end[1]] = 1

print(f"Start point in maze matrix: {maze_matrix[start[0], start[1]]}")
print(f"End point in maze matrix: {maze_matrix[end[0], end[1]]}")

# BFS Maze Solver Function
def bfs_solver(maze, start, end):
    rows, cols = maze.shape
    visited = np.zeros_like(maze, dtype=bool)
    queue = deque([(start, [start])])  # (current_position, path_taken)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while queue:
        (current, path) = queue.popleft()
        if current == end:
            return path

        if visited[current]:
            continue

        visited[current] = True

        for direction in directions:
            next_row = current[0] + direction[0]
            next_col = current[1] + direction[1]
            next_pos = (next_row, next_col)

            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and not visited[next_pos]
                and maze[next_pos] == 1
            ):
                queue.append((next_pos, path + [next_pos]))

    return None  # No solution found

# Solve the maze using BFS
bfs_path = bfs_solver(maze_matrix, start, end)

# Check if a path was found and display the results
if bfs_path:
    print(f"BFS Path found: {bfs_path}")

    # Draw the path on the original maze image
    solved_maze = maze_image.copy()
    for (row, col) in bfs_path:
        solved_maze[row, col] = [0, 0, 255]  # Mark path in red

    # Display the solved maze
    cv2.imshow("Solved Maze", solved_maze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the solved maze to a file
    cv2.imwrite("solved_maze_bfs.jpg", solved_maze)
else:
    print("No solution found.")