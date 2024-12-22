import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import pandas as pd
import matplotlib.pyplot as plt

# URLs of the Flask backends
API_BOX_URL = "http://127.0.0.1:5000/detect-boxes"
API_MAZE_URL = "http://127.0.0.1:5000/solve-maze"

# Page config
st.set_page_config(page_title="Detection and Maze Solver", layout="wide")

# Title and description
st.title("Detection and Maze Solver System")
st.markdown("""
This application provides two main functionalities:
- Box Detection: Detects boxes in images and provides information about their positions, dimensions, and orientations.
- Maze Solver: Solves maze images using algorithms like BFS, Dijkstra, and A*.
You can switch between them using the navigation bar below.
""")

# Navigation menu
app_mode = st.sidebar.selectbox("Choose the Section", options=["Box Detection", "Maze Solver"])

# Box Detection Section
if app_mode == "Box Detection":
    st.markdown("""
    ### Box Detection
    This application detects boxes in images and provides detailed information about their:
    - Position (x, y coordinates)
    - Dimensions (width and height)
    - Orientation (angle)
    """)

    # File uploader for box detection
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image button
        if st.button("Detect Boxes"):
            st.subheader("Processing...")

            with st.spinner("Analyzing image..."):
                try:
                    # Prepare the image file for the POST request
                    files = {"image": ("image.png", uploaded_file.getvalue(), "image/png")}

                    # Make POST request to Flask API
                    response = requests.post(API_BOX_URL, files=files)

                    if response.status_code == 200:
                        data = response.json()
                        detections = data["detections"]
                        processing_time = data["processing_time"]
                        num_boxes = data["num_boxes"]

                        # Convert hex string back to bytes and create image
                        annotated_bytes = bytes.fromhex(data["annotated_image"])
                        annotated_image = Image.open(io.BytesIO(annotated_bytes))

                        # Display results
                        with col2:
                            st.subheader("Detected Boxes")
                            st.image(annotated_image, caption="Detected Boxes", use_column_width=True)

                        # Display metrics
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Number of Boxes", num_boxes)
                        with metrics_col2:
                            st.metric("Processing Time", f"{processing_time:.2f} seconds")

                        # Display detailed results
                        st.subheader("Detection Details")
                        if detections:
                            # Convert detections to DataFrame
                            df = pd.DataFrame(detections)
                            st.dataframe(df)

                            # Download button for CSV
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="box_detections.csv",
                                mime="text/csv"
                            )

                            # Create scatter plot of box positions
                            st.subheader("Box Position Visualization")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(df['x'], df['y'], 
                                              c=df['orientation'], 
                                              cmap='viridis', 
                                              s=100)
                            plt.colorbar(scatter, label='Orientation (degrees)')
                            ax.set_xlabel('X Position')
                            ax.set_ylabel('Y Position')
                            ax.set_title('Box Positions and Orientations')
                            st.pyplot(fig)

                        else:
                            st.warning("No boxes detected in the image.")
                    else:
                        st.error(f"Error: {response.json()['error']}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Add sidebar with information for box detection
    with st.sidebar:
        st.header("About Box Detection")
        st.markdown("""
        This application uses computer vision techniques to:
        1. Detect rectangular objects in images
        2. Calculate their positions and dimensions
        3. Determine their orientation
        4. Provide downloadable results

        Upload an image to get started!
        """)

# Maze Solver Section
elif app_mode == "Maze Solver":
    st.markdown("""
    ### Maze Solver
    This application solves maze images using different algorithms:
    - BFS (Breadth-First Search)
    - Dijkstra's Algorithm
    - A* (A Star) Algorithm
    """)

    # Sidebar for maze solver algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm", options=["BFS", "Dijkstra", "A*"]
    )

    # File uploader for maze solver
    uploaded_file = st.file_uploader("Upload Maze Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded maze image
        st.subheader("Uploaded Maze Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Maze Image", use_column_width=True)

        # Solve maze button
        if st.button("Solve Maze"):
            st.subheader("Solving...")

            # Send image and algorithm to the Flask backend
            with st.spinner("Processing..."):
                try:
                    # Convert image to bytes
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format="PNG")
                    image_bytes = image_bytes.getvalue()

                    # Prepare the payload for the POST request
                    files = {"maze_image": ("maze.png", image_bytes, "image/png")}
                    data = {"algorithm": algorithm}

                    # Make a POST request to the Flask API
                    response = requests.post(
                        API_MAZE_URL,
                        data=data,
                        files=files,
                    )

                    # Parse response
                    if response.status_code == 200:
                        data = response.json()
                        path_x, path_y = data["path_x"], data["path_y"]

                        # Plot results
                        st.subheader(f"Path Found Using {algorithm}")
                        skeleton_image = np.array(image.convert("L"))
                        plt.figure(figsize=(10, 6))
                        plt.imshow(skeleton_image, cmap="gray")
                        plt.plot(path_x, path_y, color="red", linewidth=2)
                        plt.title(f"Path Found Using {algorithm}")
                        plt.axis("off")
                        st.pyplot(plt)
                    else:
                        st.error(f"Error: {response.json()['error']}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with st.sidebar:
        st.header("About Maze Solver")
        st.markdown("""
        This application uses different algorithms to solve maze images:
        - BFS: Breadth-First Search
        - Dijkstra: Shortest path algorithm
        - A*: A* search algorithm (with heuristics)

        Upload a maze image to get started!
        """)