from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

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

def process_image(image_bytes, target_size=(640, 640)):
    """Process uploaded image bytes"""
    image = Image.open(io.BytesIO(image_bytes))
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return cv2.resize(cv_image, target_size, interpolation=cv2.INTER_LINEAR)

def create_annotated_image(image, detections):
    """Create annotated image with box detections"""
    annotated = image.copy()
    for detection in detections:
        x, y = int(detection["x"]), int(detection["y"])
        w, h = detection["width"], detection["height"]
        angle = detection["orientation"]
        
        # Draw bounding box and center
        box_points = cv2.boxPoints(((x, y), (w, h), angle))
        box_points = np.int32(box_points)
        cv2.drawContours(annotated, [box_points], 0, (0, 255, 0), 2)
        cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
        
        # Add orientation label
        cv2.putText(annotated, f"Angle: {angle}°", 
                   (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
    
    return annotated

@app.route('/detect-boxes', methods=['POST'])
def detect_boxes_endpoint():
    try:
        # Get the image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Process image and detect boxes
        start_time = time.time()
        
        processed_image = process_image(image_bytes)
        detections = detect_boxes(processed_image)
        
        # Create annotated image
        annotated_image = create_annotated_image(processed_image, detections)
        
        # Convert annotated image to bytes
        _, buffer = cv2.imencode('.png', annotated_image)
        annotated_bytes = buffer.tobytes()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'detections': detections,
            'annotated_image': annotated_bytes.hex(),  # Convert bytes to hex string
            'processing_time': processing_time,
            'num_boxes': len(detections)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)