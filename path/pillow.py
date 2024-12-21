from PIL import Image

def image_to_binary(image_path):
  """
  Converts a grayscale image to a binary representation.

  Args:
    image_path: Path to the image file.

  Returns:
    A list of lists, where each inner list represents a row of pixels, 
    and each element is 1 for black and 0 for white.
  """

  try:
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    width, height = img.size

    binary_image = []
    for y in range(height):
      row = []
      for x in range(width):
        pixel = img.getpixel((x, y))
        row.append(1 if pixel < 128 else 0)  # Threshold at 128 for black/white
      binary_image.append(row)

    return binary_image

  except FileNotFoundError:
    print(f"Error: Image file not found at '{image_path}'.")
    return None

# Example usage
image_path = r"C:\Users\Lenovo\OneDrive\Desktop\Tech-A-Thon_Tech4Stack\testing data\maze.jpg"
binary_maze = image_to_binary(image_path)

if binary_maze:
  # Print a sample of the binary representation
  for row in binary_maze[:10]:  # Print first 10 rows
    print(row)
