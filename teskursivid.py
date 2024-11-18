# Import Dependencies
from ultralytics import YOLO
import cv2
import os

# Load YOLO Model
model = YOLO("kursiv1.pt")  # Use your custom trained weights

# Path to the input video
video_path = "objek/b2.mp4"  # Replace with the path to your video file

# Check if the video exists
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Get video details
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Resize dimensions (e.g., 640x360 for smaller output)
output_width = 640
output_height = 360
output_fps = max(15, input_fps // 2)  # Reduce FPS to half or minimum 15 FPS

# Output video settings
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "detected_kursi_compressed.mp4")
fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264 codec for smaller size
out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (output_width, output_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to smaller dimensions
    resized_frame = cv2.resize(frame, (output_width, output_height))

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

    # Write the processed frame to the output video
    out.write(resized_frame)

    # Display the frame with detections
    cv2.imshow("Chair Detection", resized_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to {output_video_path}")
