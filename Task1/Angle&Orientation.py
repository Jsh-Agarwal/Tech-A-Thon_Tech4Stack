import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageTk
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor  # For parallel processing

# List of image paths (update paths with actual image filenames)
image_paths = ['obj1.jpg', 'obj2.jpg', 'obj3.jpg', 'obj4.jpg']  # Replace with your image filenames

def create_output_structure():
    """Create organized output directory structure"""
    base_dir = "output"
    subdirs = ["preprocessed", "csv_files", "annotated", "text_reports"]
    
    for subdir in subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
    
    return base_dir

def generate_text_report(image_path, detection_results, processing_time, output_dir):
    """Generate detailed text report for each image"""
    basename = os.path.basename(image_path)
    report_path = os.path.join(output_dir, "text_reports", f"{basename}_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Analysis Report for {basename}\n")
        f.write("="* 50 + "\n\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds\n")
        f.write(f"Number of Boxes Detected: {len(detection_results)}\n\n")
        
        f.write("Detailed Box Information:\n")
        f.write("-"* 30 + "\n")
        for idx, box in enumerate(detection_results, 1):
            f.write(f"\nBox #{idx}:\n")
            f.write(f"  Center Position: ({box['x']:.2f}, {box['y']:.2f})\n")
            f.write(f"  Dimensions: {box['width']}x{box['height']} pixels\n")

def preprocess_images(image_paths, target_size=(640, 640), output_dir="preprocessed_images"):
    """Preprocess images by resizing and saving them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_images = []
    
    for idx, path in enumerate(image_paths):
        image = Image.open(path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(cv_image, target_size, interpolation=cv2.INTER_LINEAR)
        preprocessed_images.append((resized_image, path))

        # Save preprocessed image
        output_path = os.path.join(output_dir, f"preprocessed_image_{idx+1}.jpg")
        cv2.imwrite(output_path, resized_image)

    return preprocessed_images

def detect_boxes(image):
    """Detect boxes in the image using OpenCV."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:  # Detecting quadrilateral (box)
            x, y, w, h = cv2.boundingRect(approx)
            center_x = x + w / 2
            center_y = y + h / 2
            angle = cv2.minAreaRect(contour)[-1]  # Get the angle of rotation
            
            boxes.append({
                "x": round(center_x, 2),
                "y": round(center_y, 2),
                "z": 0,  # Placeholder for depth information
                "orientation": round(angle, 2),  # Angle of the rotated bounding box
                "width": w,
                "height": h
            })
    return boxes

def run_box_detection(images):
    """Run box detection on a list of images in parallel."""
    with ThreadPoolExecutor() as executor:
        detection_results = list(executor.map(lambda image: detect_boxes(image[0]), images))
    return detection_results

def save_detections_to_csv(detection_results, images, output_dir="detection_csv"):
    os.makedirs(output_dir, exist_ok=True)
    csv_paths = []
    for idx, (detections, (_, image_path)) in enumerate(zip(detection_results, images)):
        df = pd.DataFrame(detections)
        csv_path = os.path.join(output_dir, f"detections_{os.path.basename(image_path)}.csv")
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
    return csv_paths

def overlay_detections_on_images(images, detection_results, output_dir="annotated_images"):
    os.makedirs(output_dir, exist_ok=True)
    annotated_paths = []
    for idx, ((image, image_path), detections) in enumerate(zip(images, detection_results)):
        for detection in detections:
            x, y = int(detection["x"]), int(detection["y"])
            w, h = detection["width"], detection["height"]
            angle = detection["orientation"]
            
            # Draw the bounding box and the center
            box_points = cv2.boxPoints(((x, y), (w, h), angle))
            box_points = np.int32(box_points)
            cv2.drawContours(image, [box_points], 0, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # Add orientation label
            cv2.putText(image, f"Angle: {angle}°", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, image)
        annotated_paths.append(output_path)

    return annotated_paths

def display_images_tkinter(images):
    root = tk.Tk()
    root.title("Image Viewer")

    def show_image(index):
        if 0 <= index < len(images):
            image, path = images[index]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)

            panel.configure(image=imgtk)
            panel.image = imgtk
            label.configure(text=f"Image {index + 1}/{len(images)}: {os.path.basename(path)}")

            root.current_index = index

    def next_image():
        show_image(root.current_index + 1)

    def prev_image():
        show_image(root.current_index - 1)

    panel = tk.Label(root)
    panel.pack()

    label = tk.Label(root, text="", font=("Arial", 12))
    label.pack()

    btn_prev = tk.Button(root, text="Previous", command=prev_image)
    btn_prev.pack(side="left")

    btn_next = tk.Button(root, text="Next", command=next_image)
    btn_next.pack(side="right")

    root.current_index = 0
    show_image(root.current_index)

    root.mainloop()

# Main pipeline
if __name__ == "__main__":
    # Create output directories
    output_dir = create_output_structure()
    
    # Parallel processing pipeline
    preprocessed_images = preprocess_images(image_paths, output_dir=os.path.join(output_dir, "preprocessed"))
    detection_results = run_box_detection(preprocessed_images)
    csv_paths = save_detections_to_csv(detection_results, preprocessed_images, 
                                     output_dir=os.path.join(output_dir, "csv_files"))
    annotated_paths = overlay_detections_on_images(preprocessed_images, detection_results, 
                                                 output_dir=os.path.join(output_dir, "annotated"))
    
    # Generate text reports
    for image, detections in zip(preprocessed_images, detection_results):
        start_time = time.time()
        generate_text_report(image[1], detections, time.time() - start_time, output_dir)
    
    # Display results
    display_images_tkinter(preprocessed_images)
