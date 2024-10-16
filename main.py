import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image

def remove_background(image, num_clusters=5, tolerance=40):
    # Convert image to NumPy array
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    # Check if the image has 4 channels (RGBA) or 3 channels (RGB)
    if image_np.shape[2] == 3:  # If RGB, add an alpha channel
        image_np = np.dstack([image_np, np.full((height, width), 255)])  # Add opaque alpha channel

    # Reshape the image to a 2D array of pixels (including alpha channel)
    pixels = image_np.reshape((-1, 4))

    # Apply KMeans clustering to find dominant colors
    kClust = KMeans(n_clusters=num_clusters, random_state=42)
    kClust.fit(pixels)

    # Extract dominant color (background)
    dominant_colors = kClust.cluster_centers_.astype(int)
    counts = np.bincount(kClust.labels_)
    background_color = dominant_colors[np.argmax(counts)]

    # Create masks for the background
    lower = np.maximum(background_color[:3] - tolerance, 0)  # Only RGB for masking
    upper = np.minimum(background_color[:3] + tolerance, 255)

    # Masking background based on tolerance
    mask = cv2.inRange(image_np[:, :, :3], lower, upper)  # Apply to RGB channels only

    # Set masked area to transparent
    image_np[mask != 0] = [0, 0, 0, 0]  # Set background to fully transparent (RGBA)

    # Convert to uint8 format
    image_np = image_np.astype(np.uint8)

    # Convert back to PIL image and return
    return Image.fromarray(image_np)


# Streamlit UI
st.title("Background Removal with K-Means Clustering")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image from uploader
    image = Image.open(uploaded_file)

    st.image(image, caption="Original Image", use_column_width=True)

    num_clusters = st.slider("Number of Clusters", 2, 10, 5)
    tolerance = st.slider("Tolerance", 0, 100, 40)

    result_image = remove_background(image, num_clusters=num_clusters, tolerance=tolerance)

    st.image(result_image, caption="Processed Image with Background Removed", use_column_width=True)

    st.download_button("Download Processed Image", result_image.tobytes(), file_name="processed_image.png",
                       mime="image/png")
