import cv2
import numpy as np
from typing import List, Tuple, Dict
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor

class BoxDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        """
        Initialize the box detector with camera calibration parameters.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.eye(3)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((1, 5))
        
        # ArUco dictionary for marker detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Parameters for box detection
        self.min_box_area = 500  # Lowered minimum area
        self.canny_thresholds = (50, 150)
        
        # Box counting statistics
        self.total_boxes = 0
        self.boxes_with_aruco = 0
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for better box detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological close to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def detect_boxes(self, image: np.ndarray) -> Tuple[List[Dict], float, Dict]:
        """
        Detect boxes in the image and return their properties.
        
        Args:
            image: Input image
            
        Returns:
            Tuple containing:
            - List of dictionaries containing box properties
            - Processing time
            - Dictionary with box statistics
        """
        start_time = time.time()
        
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Find contours
        edges = cv2.Canny(processed, *self.canny_thresholds)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        print(f"Number of contours found: {len(contours)}")
        
        boxes = []
        self.total_boxes = 0
        self.boxes_with_aruco = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"Contour area: {area}")
            
            # Filter small contours
            if area < self.min_box_area:
                continue
                
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the polygon has 4 vertices (is rectangular)
            if len(approx) == 4:
                self.total_boxes += 1
                
                # Get box properties
                rect = cv2.minAreaRect(contour)
                box_center = rect[0]
                box_size = rect[1]
                box_angle = rect[2]
                
                # Get 3D position (assuming known box dimensions and camera parameters)
                x, y = box_center
                z = self.estimate_depth(box_size)
                
                boxes.append({
                    'position': (x, y, z),
                    'orientation': box_angle,
                    'size': box_size,
                    'contour': contour
                })
        
        # Convert image to grayscale for ArUco detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            # Try using the newer ArUco API first
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Fallback to older API if newer one is not available
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        # Associate ArUco markers with detected boxes
        if ids is not None:
            self._associate_aruco_with_boxes(boxes, corners, ids)
        
        processing_time = time.time() - start_time
        
        # Compile box statistics
        stats = {
            'total_boxes': self.total_boxes,
            'boxes_with_aruco': self.boxes_with_aruco,
            'boxes_without_aruco': self.total_boxes - self.boxes_with_aruco
        }
        
        return boxes, processing_time, stats

    def estimate_depth(self, box_size: Tuple[float, float]) -> float:
        """
        Estimate the depth (Z coordinate) of a box based on its apparent size.
        This is a simplified calculation - in practice, you'd need proper 3D reconstruction.
        
        Args:
            box_size: Apparent size of the box in the image
            
        Returns:
            Estimated depth in centimeters
        """
        # This is a simplified calculation assuming a known real box size
        KNOWN_BOX_WIDTH = 50  # cm
        focal_length = self.camera_matrix[0, 0]
        
        # Use similar triangles to estimate depth
        depth = (KNOWN_BOX_WIDTH * focal_length) / box_size[0]
        return depth
    
    def _associate_aruco_with_boxes(
        self, boxes: List[Dict], 
        aruco_corners: List[np.ndarray], 
        aruco_ids: np.ndarray
    ) -> None:
        """
        Associate detected ArUco markers with boxes.
        
        Args:
            boxes: List of detected boxes
            aruco_corners: Corners of detected ArUco markers
            aruco_ids: IDs of detected ArUco markers
        """
        for box in boxes:
            box_contour = box['contour']
            for corners, id_ in zip(aruco_corners, aruco_ids):
                marker_center = np.mean(corners[0], axis=0)
                if cv2.pointPolygonTest(box_contour, tuple(marker_center), False) >= 0:
                    box['aruco_id'] = id_[0]
                    self.boxes_with_aruco += 1
                    break
    
    def visualize_results(
        self, image: np.ndarray, 
        boxes: List[Dict], 
        processing_time: float,
        stats: Dict,
        image_name: str
    ) -> np.ndarray:
        """
        Visualize the detection results on the image.
        
        Args:
            image: Original image
            boxes: List of detected boxes
            processing_time: Time taken for detection
            stats: Dictionary containing box statistics
            image_name: Name of the input image file
            
        Returns:
            Annotated image
        """
        result = image.copy()
        
        # Draw image name
        cv2.putText(
            result, f"Image: {image_name}", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        
        # Draw box statistics
        cv2.putText(
            result, f"Total Boxes: {stats['total_boxes']}", 
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            result, f"With ArUco: {stats['boxes_with_aruco']}", 
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            result, f"Without ArUco: {stats['boxes_without_aruco']}", 
            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        for box in boxes:
            # Draw box contour
            cv2.drawContours(result, [box['contour']], 0, (0, 255, 0), 2)
            
            # Draw position and orientation
            x, y, z = box['position']
            cv2.putText(
                result, f"({x:.1f}, {y:.1f}, {z:.1f}cm)", 
                (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # Draw ArUco ID if available
            if 'aruco_id' in box:
                cv2.putText(
                    result, f"ID: {box['aruco_id']}", 
                    (int(x), int(y) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
        
        # Add processing time
        cv2.putText(
            result, f"Processing time: {processing_time:.3f}s", 
            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        return result

def preprocess_images(image_paths, target_size=(640, 640), output_dir="preprocessed_images"):
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_images = []
    for idx, path in enumerate(image_paths):
        image = Image.open(path)
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        resized_image = cv2.resize(cv_image, target_size, interpolation=cv2.INTER_LINEAR)
        preprocessed_images.append((resized_image, path))
        output_path = os.path.join(output_dir, f"preprocessed_image_{idx+1}.jpg")
        cv2.imwrite(output_path, resized_image)
    return preprocessed_images

def detect_boxes_basic(image: np.ndarray) -> List[Dict]:
    """
    Basic box detection using edge detection and contour approximation.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        List of dictionaries containing box properties:
        - x, y: center coordinates
        - z: depth (set to 0)
        - orientation: angle in degrees
        - width, height: box dimensions
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            center_x = x + w / 2
            center_y = y + h / 2
            angle = cv2.minAreaRect(contour)[-1]
            boxes.append({
                "x": round(center_x, 2),
                "y": round(center_y, 2),
                "z": 0,
                "orientation": round(angle, 2),
                "width": w,
                "height": h
            })
    return boxes

def detect_boxes_contour(image_path: str, output_folder="Output") -> int:
    """
    Advanced box detection using contour filtering and area-based thresholding.
    Saves annotated images and detection counts to output folder.
    
    Args:
        image_path: Path to input image
        output_folder: Directory to save results
        
    Returns:
        Number of boxes detected
        
    Outputs:
        - Annotated image with detected boxes
        - Text file with box count
    """
    os.makedirs(output_folder, exist_ok=True)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 30, 150, 3)
    dilated = cv2.dilate(canny, np.ones((5, 5), np.uint8), iterations=1)
    cnt, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter contours by area
    contour_areas = [cv2.contourArea(c) for c in cnt]
    if contour_areas:
        average_area = np.mean(contour_areas)
        min_area = average_area / 2
        cnt = [c for c in cnt if cv2.contourArea(c) >= min_area]

    # Save results
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(output_image, cnt, -1, (0, 255, 0), 2)
    outfile = os.path.join(output_folder, f"contour_detect_{os.path.basename(image_path)}")
    cv2.imwrite(outfile, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    
    boxes_count = len(cnt)
    count_file = os.path.join(output_folder, f"box_count_{os.path.basename(image_path)}.txt")
    with open(count_file, 'w') as f:
        f.write(f"Number of boxes detected: {boxes_count}")
    
    print(f"Image: {image_path}")
    print(f"Number of boxes detected: {boxes_count}")
    print("-" * 50)
    return boxes_count

def read_box_counts(output_folder="Output"):
    """Read and display box counts from all output files"""
    for filename in os.listdir(output_folder):
        if filename.startswith("box_count_") and filename.endswith(".txt"):
            with open(os.path.join(output_folder, filename), 'r') as f:
                print(f"File: {filename}")
                print(f"{f.read()}")
                print("-" * 50)

def run_box_detection(images):
    detection_results = []
    with ThreadPoolExecutor() as executor:
        detection_results = list(executor.map(lambda image: detect_boxes_basic(image[0]), images))
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
    for (image, image_path), detections in zip(images, detection_results):
        for detection in detections:
            x, y = int(detection["x"]), int(detection["y"])
            w, h = detection["width"], detection["height"]
            angle = detection["orientation"]
            box_points = cv2.boxPoints(((x, y), (w, h), angle))
            box_points = np.int32(box_points)
            cv2.drawContours(image, [box_points], 0, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f"Angle: {angle}°", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

def save_detection_results(image_path: str, boxes: List[Dict], stats: Dict, 
                         processing_time: float, output_base="Output") -> None:
    """
    Save comprehensive detection results for each image.
    
    Args:
        image_path: Path to the original image
        boxes: List of detected boxes and their properties
        stats: Detection statistics
        processing_time: Time taken for detection
        output_base: Base output directory
    """
    # Create organized folder structure
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]
    output_dir = os.path.join(output_base, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results to text file
    results_file = os.path.join(output_dir, f"{base_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Detection Results for {image_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Boxes Detected: {stats['total_boxes']}\n")
        f.write(f"Boxes with ArUco: {stats['boxes_with_aruco']}\n")
        f.write(f"Boxes without ArUco: {stats['boxes_without_aruco']}\n")
        f.write(f"Processing Time: {processing_time:.3f} seconds\n\n")
        
        # Individual box details
        f.write("Individual Box Details:\n")
        f.write("-" * 20 + "\n")
        for idx, box in enumerate(boxes, 1):
            f.write(f"\nBox #{idx}:\n")
            x, y, z = box['position']
            f.write(f"Position: ({x:.1f}, {y:.1f}, {z:.1f}cm)\n")
            f.write(f"Orientation: {box['orientation']:.2f}°\n")
            f.write(f"Size: {box['size'][0]:.1f} x {box['size'][1]:.1f}\n")
            if 'aruco_id' in box:
                f.write(f"ArUco ID: {box['aruco_id']}\n")

# Update main block to use the new function
if __name__ == "__main__":
    image_paths = ['obj1.jpg', 'obj2.jpg', 'obj3.jpg', 'obj4.jpg']
    
    # Create main output directory
    output_base = "Detection_Output"
    os.makedirs(output_base, exist_ok=True)
    
    # Process each image
    detector = BoxDetector()
    for image_path in image_paths:
        if os.path.exists(image_path):
            # Read and process image
            image = cv2.imread(image_path)
            boxes, processing_time, stats = detector.detect_boxes(image)
            
            # Save comprehensive results
            save_detection_results(image_path, boxes, stats, processing_time, output_base)
            
            # Save annotated image
            result = detector.visualize_results(image, boxes, processing_time, stats, image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_base, base_name, f"{base_name}_annotated.jpg"), result)
            
            # Also run contour detection
            detect_boxes_contour(image_path, os.path.join(output_base, base_name))
        else:
            print(f"Warning: Image file {image_path} not found")
    
    print("\nProcessing complete. Results saved in:", output_base)