from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# YOLO model configuration
YOLO_CONFIG = "yolov3.cfg"  # Path to YOLO config file
YOLO_WEIGHTS = "yolov3-tiny.weights"  # Path to YOLO weights file
COCO_NAMES = "coco.names"  # Path to COCO names file

# Load YOLO
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Use GPU if available
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load COCO class names
with open(COCO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/detect', methods=['POST'])
def detect_objects():
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Prepare image for YOLO
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Analyze detections
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for detection
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detections.append({"label": label, "confidence": confidences[i], "box": [x, y, w, h]})

    # Send results via WebSocket
    socketio.emit('detection_result', detections)

    return jsonify({"detections": detections})

# WebSocket for live updates
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app, debug=True)