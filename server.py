from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class labels
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

# Colors for drawing
colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

@app.route("/process-image/", methods=["POST"])
def process_image():
    # Read image from the uploaded file
    file = request.files['file']
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width, channels = img.shape

    # Image center
    image_center_x = width // 2
    image_center_y = height // 2

    # Determine scaling factor based on image shape
    if width == height:
        scale_x = 5 / (width // 2)
        scale_y = 5 / (height // 2)
    else:
        scale_x = 10 / (width // 2)
        scale_y = 5 / (height // 2)

    # Prepare the image for YOLO
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
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression to reduce overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Find box with highest confidence
    coordinates = {}
    if len(indexes) > 0:
        max_conf_index = np.argmax(confidences)
        if max_conf_index in indexes:
            x, y, w, h = boxes[max_conf_index]
            label = str(classes[class_ids[max_conf_index]])

            # Calculate center point of the detected object
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate new coordinate system relative to image center
            relative_x = (center_x - image_center_x) * scale_x
            relative_y = (image_center_y - center_y) * scale_y

            # Draw the bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)

            # Display the scaled coordinates on the top-left corner
            coord_text = f"({relative_x:.2f}, {relative_y:.2f})"
            cv2.putText(img, coord_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, colorRed, 2)

            coordinates = {"x": round(relative_x, 2), "y": round(relative_y, 2)}

    # Save output image
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)

    return jsonify({"coordinates": coordinates, "image_url": "/download-image/"})

@app.route("/download-image/", methods=["GET"])
def download_image():
    return send_file("output.jpg", mimetype='image/jpeg')

if __name__ == "__main__":
    print("Flask server running...")
    app.run(host='0.0.0.0', port=5000)
