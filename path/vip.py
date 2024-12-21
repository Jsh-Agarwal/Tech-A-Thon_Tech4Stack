import cv2  # OpenCV library for image processing
import numpy as np  # NumPy for matrix manipulations
from queue import PriorityQueue  # PriorityQueue for A* algorithm

# Load and preprocess the maze image
def preprocess_maze(image_path):
    print(f"Attempting to read image from: {image_path}")  # Debugging line
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # If the image is not found, raise an error
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Threshold the image to get a binary representation
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

# Convert maze image to matrix
def image_to_matrix(binary_image):
    return (binary_image // 255).astype(np.uint8)

# Find the start and end points (red and green markers)
def find_start_and_end(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green markers in HSV
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([70, 255, 255])

    # Create masks for red and green
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Get coordinates of red (start) and green (end)
    start = np.column_stack(np.where(red_mask > 0))
    end = np.column_stack(np.where(green_mask > 0))

    if start.size == 0 or end.size == 0:
        raise ValueError("Start or end point not found in the image.")
    
    # Return the first detected start and end points
    return tuple(start[0]), tuple(end[0])

# A* Algorithm
def a_star(maze, start, end):
    rows, cols = maze.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while not open_set.empty():
        _, current = open_set.get()
        if current == end:
            print("Path found!")
            return reconstruct_path(came_from, current)
        
        neighbors = get_neighbors(current, rows, cols)
        print(f"Current: {current}, Neighbors: {neighbors}")
        for neighbor in neighbors:
            if maze[neighbor] == 0:  # Skip walls
                continue
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                open_set.put((f_score[neighbor], neighbor))
    print("No path found.")
    return []

# Helper functions for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(node, rows, cols):
    x, y = node
    neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < rows and 0 <= ny < cols]

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

# Highlight the path on the maze
def draw_path(image, path):
    for x, y in path:
        image[x, y] = [0, 0, 255]  # Red for the path
    return image

# Main function
def solve_maze(image_path):
    # Preprocess the maze and get the binary representation
    binary_image = preprocess_maze(image_path)
    
    # Convert the binary image into a matrix
    maze_matrix = image_to_matrix(binary_image)
    
    # Find the start and end points in the maze
    start, end = find_start_and_end(image_path)
    print(f"Start: {start}, End: {end}")  # Debugging output

    # Run the A* algorithm to find the shortest path
    path = a_star(maze_matrix, start, end)
    
    if path:
        # If a path is found, highlight it on the original image
        original_image = cv2.imread(image_path)
        solved_maze = draw_path(original_image, path)
        
        # Display the solved maze
        cv2.imshow("Solved Maze", solved_maze)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite("solved_maze.jpg", solved_maze)
    else:
        print("No solution found.")

image_path = r'C:\Users\Lenovo\OneDrive\Desktop\Tech-A-Thon_Tech4Stack\testing data\maze.jpg'

solve_maze(image_path)


# 810,850