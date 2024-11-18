# Import Dependencies
from ultralytics import YOLO
import cv2
import os

# Load YOLO Model
model = YOLO("kursiv1.pt")  # Use your custom trained weights

# Path to the input image
image_path = "objek/tes2.png"  # Replace with the path to your image file

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit()

# Load and preprocess the image
img = cv2.imread(image_path)
if img is None:
    print("Error: Could not read the selected image.")
    exit()

# Resize the image to a fixed size (e.g., 640x640)
resize_width = 640
resize_height = 640
resized_img = cv2.resize(img, (resize_width, resize_height))

# Perform object detection
results = model(resized_img)

# Draw results on the resized image
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
        cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 2)

        # Add label with confidence
        label = f"{class_name}: {confidence}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y1 = max(y1 - 10, 0)
        label_y2 = label_y1 + label_size[1]
        cv2.rectangle(resized_img, (x1, label_y1), (x1 + label_size[0], label_y2), color, -1)  # Background for label
        cv2.putText(resized_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black text

# Save the resized and processed image
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_image_path = os.path.join(output_folder, "detected_kursi.png")
cv2.imwrite(output_image_path, resized_img)
print(f"Processed image saved to {output_image_path}")

# Display the image with detections
cv2.imshow("Chair Detection", resized_img)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
