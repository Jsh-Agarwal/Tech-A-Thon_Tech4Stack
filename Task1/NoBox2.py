import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

class BoxDetector:
    def __init__(self):
        self.model = self._initialize_model()
        self.tile_sizes = [(640, 640), (800, 800), (1024, 1024)]
        self.overlap = 0.3  # 30% overlap between tiles
        
    def _initialize_model(self):
        model = YOLO('yolov8n.pt')
        model.conf = 0.15  # Lower confidence threshold
        model.iou = 0.35   # Lower IOU threshold
        model.max_det = 300
        return model

    def enhance_image(self, image):
        """Multi-stage image enhancement"""
        # Convert to LAB and enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with multiple tile sizes
        clahe_small = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_large = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        l1 = clahe_small.apply(l)
        l2 = clahe_large.apply(l)
        l = cv2.addWeighted(l1, 0.5, l2, 0.5, 0)
        
        enhanced_lab = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def create_tiles(self, image, tile_size):
        """Create overlapping tiles from image"""
        height, width = image.shape[:2]
        tile_height, tile_width = tile_size
        
        stride_h = int(tile_height * (1 - self.overlap))
        stride_w = int(tile_width * (1 - self.overlap))
        
        tiles = []
        positions = []
        
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                x2 = min(x + tile_width, width)
                y2 = min(y + tile_height, height)
                x1 = max(0, x2 - tile_width)
                y1 = max(0, y2 - tile_height)
                
                tile = image[y1:y2, x1:x2]
                tiles.append(tile)
                positions.append((x1, y1, x2, y2))
        
        return tiles, positions

    def detect_boxes_in_tile(self, tile, tile_pos):
        """Detect boxes in a single tile"""
        results = self.model(tile, augment=True, verbose=False)
        detections = []
        
        x_offset, y_offset = tile_pos[0], tile_pos[1]
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Adjust coordinates to original image space
                x1, x2 = x1 + x_offset, x2 + x_offset
                y1, y2 = y1 + y_offset, y2 + y_offset
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': float(box.conf)
                })
        
        return detections

    def merge_detections(self, all_detections, iou_threshold=0.45):
        """Merge detections from different tiles"""
        if not all_detections:
            return []
            
        # Sort by confidence
        all_detections = sorted(all_detections, key=lambda x: x['conf'], reverse=True)
        merged_detections = []
        
        while all_detections:
            best = all_detections.pop(0)
            merged_detections.append(best)
            
            # Filter out overlapping boxes
            remaining_detections = []
            for det in all_detections:
                iou = self.calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining_detections.append(det)
                elif iou >= iou_threshold and det['conf'] > best['conf']:
                    # Keep the higher confidence detection
                    merged_detections[-1] = det
                    
            all_detections = remaining_detections
        
        return merged_detections

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
        """Process image with tiling and ensemble detection"""
        # Read and enhance image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Enhance image
        enhanced = self.enhance_image(image)
        
        all_detections = []
        
        # Process at different scales
        scales = [0.75, 1.0, 1.25]
        for scale in scales:
            # Resize image
            height, width = enhanced.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(enhanced, (new_width, new_height))
            
            # Process each tile size
            for tile_size in self.tile_sizes:
                tiles, positions = self.create_tiles(resized, tile_size)
                
                # Detect boxes in each tile
                for tile, pos in zip(tiles, positions):
                    detections = self.detect_boxes_in_tile(tile, pos)
                    
                    # Scale coordinates back to original size if needed
                    if scale != 1.0:
                        for det in detections:
                            x1, y1, x2, y2 = det['bbox']
                            det['bbox'] = (
                                int(x1 / scale),
                                int(y1 / scale),
                                int(x2 / scale),
                                int(y2 / scale)
                            )
                    
                    all_detections.extend(detections)
        
        # Merge all detections
        final_detections = self.merge_detections(all_detections)
        
        # Draw results
        result_image = image.copy()
        for det in final_detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['conf']:.2f}"
            cv2.putText(result_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image, len(final_detections)

def main():
    detector = BoxDetector()
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