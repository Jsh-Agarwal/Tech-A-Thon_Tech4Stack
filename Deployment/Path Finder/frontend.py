import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt

# URL of the Flask backend
API_URL = "http://127.0.0.1:5000/solve-maze"

# Title
st.title("Maze Solver")

# Sidebar for user inputs
st.sidebar.header("User Input")
algorithm = st.sidebar.selectbox(
    "Select Algorithm", options=["BFS", "Dijkstra", "A*"]
)

# File uploader
uploaded_file = st.file_uploader("Upload Maze Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
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
                    API_URL,
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
