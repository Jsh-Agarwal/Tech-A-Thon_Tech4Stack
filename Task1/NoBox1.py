import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

class HybridBoxDetector:
    def __init__(self):
        self.model = self._initialize_model()
        self.tile_sizes = [(512, 512), (640, 640), (768, 768), (896, 896)]
        self.overlap = 0.5  # 50% overlap for better detection
        self.scales = [0.75, 1.0, 1.25, 1.5]
        
    def _initialize_model(self):
        model = YOLO('yolov8n.pt')
        model.conf = 0.1  # Very low initial confidence
        model.iou = 0.25  # Lower IOU for more detections
        model.max_det = 300
        return model

    def enhance_image(self, image):
        """Multi-stage image enhancement"""
        # Store original
        original = image.copy()
        
        # Enhance method 1 - CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced1 = cv2.merge((cl,a,b))
        enhanced1 = cv2.cvtColor(enhanced1, cv2.COLOR_LAB2BGR)

        # Enhance method 2 - Contrast
        alpha = 1.3
        beta = 15
        enhanced2 = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Combine enhancements
        enhanced = cv2.addWeighted(enhanced1, 0.5, enhanced2, 0.5, 0)
        
        return [original, enhanced1, enhanced2, enhanced]

    def detect_boxes_in_image(self, image, scale=1.0):
        """Detect boxes using both YOLO and traditional CV"""
        height, width = image.shape[:2]
        detections = []

        # YOLO detection
        results = self.model(image, augment=True, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if scale != 1.0:
                    x1, x2 = x1/scale, x2/scale
                    y1, y2 = y1/scale, y2/scale
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(box.conf)
                })

        # Edge-based detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                if scale != 1.0:
                    x, w = int(x/scale), int(w/scale)
                    y, h = int(y/scale), int(h/scale)
                
                # Filter by aspect ratio
                aspect_ratio = float(w)/h if h > 0 else 0
                if 0.5 <= aspect_ratio <= 2.0:
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'conf': 0.3  # Base confidence for contour detection
                    })
        
        return detections

    def process_tiles(self, image, original_size):
        """Process image in tiles"""
        height, width = image.shape[:2]
        all_detections = []
        
        for tile_size in self.tile_sizes:
            stride_h = int(tile_size[0] * (1 - self.overlap))
            stride_w = int(tile_size[1] * (1 - self.overlap))
            
            for y in range(0, height-tile_size[1]+1, stride_h):
                for x in range(0, width-tile_size[0]+1, stride_w):
                    # Extract tile
                    tile = image[y:y+tile_size[1], x:x+tile_size[0]]
                    
                    # Detect boxes in tile
                    dets = self.detect_boxes_in_image(tile)
                    
                    # Adjust coordinates
                    for det in dets:
                        x1, y1, x2, y2 = det['bbox']
                        det['bbox'] = (
                            x1 + x,
                            y1 + y,
                            x2 + x,
                            y2 + y
                        )
                    
                    all_detections.extend(dets)
        
        return all_detections

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
        """Merge overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
        merged = []
        
        while detections:
            best = detections.pop(0)
            merged.append(best)
            
            i = 0
            while i < len(detections):
                if self.calculate_iou(best['bbox'], detections[i]['bbox']) > iou_threshold:
                    # Merge boxes if highly overlapping
                    x1 = min(best['bbox'][0], detections[i]['bbox'][0])
                    y1 = min(best['bbox'][1], detections[i]['bbox'][1])
                    x2 = max(best['bbox'][2], detections[i]['bbox'][2])
                    y2 = max(best['bbox'][3], detections[i]['bbox'][3])
                    best['bbox'] = (x1, y1, x2, y2)
                    best['conf'] = max(best['conf'], detections[i]['conf'])
                    detections.pop(i)
                else:
                    i += 1
        
        return merged

    def process_image(self, image_path):
        """Main processing pipeline"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = image.shape[:2]
        all_detections = []
        
        # Process multiple enhanced versions
        enhanced_images = self.enhance_image(image)
        
        # Process each enhanced version at multiple scales
        for enhanced in enhanced_images:
            for scale in self.scales:
                # Resize image
                scaled_width = int(width * scale)
                scaled_height = int(height * scale)
                scaled = cv2.resize(enhanced, (scaled_width, scaled_height))
                
                # Process tiles
                detections = self.process_tiles(scaled, (width, height))
                
                # Scale back coordinates
                if scale != 1.0:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        det['bbox'] = (
                            int(x1/scale),
                            int(y1/scale),
                            int(x2/scale),
                            int(y2/scale)
                        )
                
                all_detections.extend(detections)
        
        # Merge all detections
        final_detections = self.merge_detections(all_detections)
        
        # Draw results
        result_image = image.copy()
        for det in final_detections:
            x1, y1, x2, y2 = det['bbox']
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['conf']:.2f}"
            cv2.putText(result_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image, len(final_detections)

def main():
    detector = HybridBoxDetector()
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