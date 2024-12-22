import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

class EnhancedBoxDetector:
    def __init__(self):
        self.model = self._initialize_model()
        self.tile_sizes = [(512, 512), (640, 640), (768, 768)]
        self.overlap = 0.4  # 40% overlap for better detection
        self.scales = [0.75, 1.0, 1.25, 1.5]
        
    def _initialize_model(self):
        model = YOLO('yolov8n.pt')
        # Lower confidence threshold to catch more potential boxes
        model.conf = 0.1
        model.iou = 0.3
        model.max_det = 300
        return model

    def enhance_image(self, image):
        """Advanced multi-stage image enhancement"""
        enhanced_images = []
        
        # Original image
        enhanced_images.append(image.copy())
        
        # CLAHE enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_large = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        l1 = clahe_small.apply(l)
        l2 = clahe_large.apply(l)
        enhanced_l = cv2.addWeighted(l1, 0.5, l2, 0.5, 0)
        enhanced1 = cv2.merge((enhanced_l, a, b))
        enhanced1 = cv2.cvtColor(enhanced1, cv2.COLOR_LAB2BGR)
        enhanced_images.append(enhanced1)
        
        # Contrast enhancement
        alpha = 1.3  # Contrast control
        beta = 10    # Brightness control
        enhanced2 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        enhanced_images.append(enhanced2)
        
        # Edge-preserved smoothing
        enhanced3 = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
        enhanced_images.append(enhanced3)
        
        return enhanced_images

    def detect_boxes_yolo(self, image):
        """YOLO-based detection"""
        results = self.model(image, augment=True, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(box.conf),
                    'method': 'yolo'
                })
        
        return detections

    def detect_boxes_cv(self, image):
        """Traditional CV-based detection"""
        detections = []
        
        # Convert to grayscale and apply preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple edge detection methods
        edges1 = cv2.Canny(blurred, 50, 150)
        edges2 = cv2.Canny(blurred, 30, 100)
        edges = cv2.addWeighted(edges1, 0.5, edges2, 0.5, 0)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and area
                aspect_ratio = float(w)/h if h > 0 else 0
                if 0.2 <= aspect_ratio <= 5.0:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    x1, y1 = np.min(box, axis=0)
                    x2, y2 = np.max(box, axis=0)
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': 0.3,  # Base confidence for CV detection
                        'method': 'cv'
                    })
        
        return detections

    def process_tile(self, tile, scale, offset_x=0, offset_y=0):
        """Process a single tile with both methods"""
        detections = []
        
        # Get detections from both methods
        yolo_dets = self.detect_boxes_yolo(tile)
        cv_dets = self.detect_boxes_cv(tile)
        
        # Combine detections
        detections.extend(yolo_dets)
        detections.extend(cv_dets)
        
        # Adjust coordinates for tile offset and scaling
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = (
                int((x1 + offset_x) / scale),
                int((y1 + offset_y) / scale),
                int((x2 + offset_x) / scale),
                int((y2 + offset_y) / scale)
            )
        
        return detections

    def calculate_iou(self, box1, box2):
        """Calculate IOU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def merge_detections(self, detections, iou_threshold=0.4):
        """Smart merging of detections"""
        if not detections:
            return []
        
        # Sort by confidence and method (prioritize YOLO slightly)
        detections = sorted(detections, key=lambda x: (x['method'] == 'yolo', x['conf']), reverse=True)
        merged = []
        
        while detections:
            best = detections.pop(0)
            
            # Find all overlapping boxes
            overlaps = []
            remaining = []
            
            for det in detections:
                iou = self.calculate_iou(best['bbox'], det['bbox'])
                if iou > iou_threshold:
                    overlaps.append(det)
                else:
                    remaining.append(det)
            
            # If we have overlaps, merge them
            if overlaps:
                # Calculate average box coordinates weighted by confidence
                total_conf = best['conf'] + sum(det['conf'] for det in overlaps)
                x1 = sum((det['bbox'][0] * det['conf']) for det in [best] + overlaps) / total_conf
                y1 = sum((det['bbox'][1] * det['conf']) for det in [best] + overlaps) / total_conf
                x2 = sum((det['bbox'][2] * det['conf']) for det in [best] + overlaps) / total_conf
                y2 = sum((det['bbox'][3] * det['conf']) for det in [best] + overlaps) / total_conf
                
                best['bbox'] = (int(x1), int(y1), int(x2), int(y2))
                best['conf'] = total_conf / (len(overlaps) + 1)  # Average confidence
            
            merged.append(best)
            detections = remaining
        
        return merged

    def calculate_iou(self, box1, box2):
        """Calculate IOU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        intersection_x1 = max(x1_1, x1_2)
        intersection_y1 = max(y1_1, y1_2)
        intersection_x2 = min(x2_1, x2_2)
        intersection_y2 = min(y2_1, y2_2)
        
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0
            
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0


    def process_image(self, image_path):
        """Main processing pipeline"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = image.shape[:2]
        all_detections = []
        
        # Get enhanced versions of the image
        enhanced_images = self.enhance_image(image)
        
        # Process each enhanced version at multiple scales
        for enhanced in enhanced_images:
            for scale in self.scales:
                # Resize image
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled = cv2.resize(enhanced, (scaled_width, scaled_height))
                
                # Process image in tiles
                for tile_size in self.tile_sizes:
                    stride_h = int(tile_size[0] * (1 - self.overlap))
                    stride_w = int(tile_size[1] * (1 - self.overlap))
                    
                    for y in range(0, scaled_height, stride_h):
                        for x in range(0, scaled_width, stride_w):
                            # Extract tile
                            x2 = min(x + tile_size[0], scaled_width)
                            y2 = min(y + tile_size[1], scaled_height)
                            tile = scaled[y:y2, x:x2]
                            
                            # Process tile
                            detections = self.process_tile(tile, scale, x, y)
                            all_detections.extend(detections)
        
        # Merge all detections
        final_detections = self.merge_detections(all_detections)
        
        # Draw results
        result_image = image.copy()
        for det in final_detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if det['method'] == 'yolo' else (255, 0, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{det['conf']:.2f}"
            cv2.putText(result_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image, len(final_detections)

def main():
    detector = EnhancedBoxDetector()
    total_boxes = 0
    
    try:
        image_files = [f for f in os.listdir('input_images') 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in image_files:
            image_path = os.path.join('input_images', filename)
            print(f"\nProcessing: {filename}")
            
            try:
                result_image, num_boxes = detector.process_image(image_path)
                
                # Save result
                output_path = os.path.join('output', f"annotated_{filename}")
                cv2.imwrite(output_path, result_image)
                
                total_boxes += num_boxes
                print(f"Detected {num_boxes} boxes in {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        print(f"\nTotal boxes detected across all images: {total_boxes}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()