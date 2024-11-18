# Import Dependencies
from ultralytics import YOLO
import cv2
import os

# Load YOLO Model
model = YOLO("kursi.pt")  # Use your custom trained weights

# Start Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 520)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Process video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    # Resize the frame to a fixed size (e.g., 640x640)
    resize_width = 640
    resize_height = 640
    resized_frame = cv2.resize(frame, (resize_width, resize_height))

    # Perform object detection
    results = model(resized_frame)

    # Draw results on the frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract confidence and class name
            confidence = round(float(box.conf[0]) * 100, 2)
            class_name = "Chair"  # Use "Chair" since the model is trained for chairs only

            # Draw bounding box
            color = (0, 255, 0)  # Green box
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)

            # Add label with confidence
            label = f"{class_name}: {confidence}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y1 = max(y1 - 10, 0)
            label_y2 = label_y1 + label_size[1]
            cv2.rectangle(resized_frame, (x1, label_y1), (x1 + label_size[0], label_y2), color, -1)  # Background for label
            cv2.putText(resized_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text

    # Display the frame with detections
    cv2.imshow("Chair Detection", resized_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
