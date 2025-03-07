from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
try:
    yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except Exception as e:
    print("Error loading YOLO model:", e)
    raise

# Load class labels
try:
    with open("coco.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
except Exception as e:
    print("Error loading coco.names:", e)
    classes = []

# Get layer names and output layers from YOLO
layer_names = yolo.getLayerNames()
# Flatten in case getUnconnectedOutLayers returns a 2D array (OpenCV version dependent)
unconnected = yolo.getUnconnectedOutLayers()
if hasattr(unconnected, "flatten"):
    unconnected = unconnected.flatten()
output_layers = [layer_names[i - 1] for i in unconnected]

# Colors for drawing bounding boxes and text
colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

@app.route('/')
def home():
    return "Flask server is running!"

@app.route("/process-image/", methods=["POST"])
def process_image():
    try:
        # Verify that a file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read image from the uploaded file into a NumPy array
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        height, width, channels = img.shape

        # Calculate image center
        image_center_x = width // 2
        image_center_y = height // 2

        # Determine scaling factors based on image dimensions
        if width == height:
            scale_x = 5 / (width // 2)
            scale_y = 5 / (height // 2)
        else:
            scale_x = 10 / (width // 2)
            scale_y = 5 / (height // 2)

        # Prepare the image for YOLO detection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo.setInput(blob)
        outputs = yolo.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Process YOLO outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x_det = int(detection[0] * width)
                    center_y_det = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x_det - w / 2)
                    y = int(center_y_det - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression to reduce overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if hasattr(indexes, "flatten"):
            indexes = indexes.flatten()
        
        coordinates = {}
        if len(indexes) > 0:
            # Select the box with the highest confidence among those kept by NMS
            max_conf_index = max(indexes, key=lambda i: confidences[i])
            x, y, w, h = boxes[max_conf_index]
            label = str(classes[class_ids[max_conf_index]])

            # Calculate center of detected object
            center_x_box = x + w // 2
            center_y_box = y + h // 2

            # Calculate relative coordinates with respect to image center
            relative_x = (center_x_box - image_center_x) * scale_x
            relative_y = (image_center_y - center_y_box) * scale_y

            # Draw bounding box and coordinate text on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)
            coord_text = f"({relative_x:.2f}, {relative_y:.2f})"
            cv2.putText(img, coord_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, colorRed, 2)

            coordinates = {"x": round(relative_x, 2), "y": round(relative_y, 2)}

        # Save the processed image to disk
        output_path = "output.jpg"
        cv2.imwrite(output_path, img)

        return jsonify({"coordinates": coordinates, "image_url": "/download-image/"})
    
    except Exception as e:
        print("Error in /process-image:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/download-image/", methods=["GET"])
def download_image():
    try:
        return send_file("output.jpg", mimetype='image/jpeg')
    except Exception as e:
        print("Error in /download-image:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Flask server running...")
    app.run(host='0.0.0.0', port=5000, debug=True)
