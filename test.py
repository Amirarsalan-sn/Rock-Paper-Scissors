import cv2
from ultralytics import YOLO

# Load the YOLO models
model_hand = YOLO('yolov11n-hands.pt')
model_face = YOLO('yolov11n-face.pt')


# Function to draw bounding boxes and labels on the image
def draw_bounding_boxes(image, detections, class_names):
    for detection in detections:
        cls = detection.cls.item()
        conf = detection.conf.item()
        x1, y1, x2, y2 = detection.xyxy[0].tolist()
        label = f"{class_names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
        cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# Load the image
image_path = 'test_4.jpg'  # Change this to your image path
image = cv2.imread(image_path)

# Predict faces and hands
face_detections = model_face.predict(source=image)  # Confidence threshold
hand_detections = model_hand.predict(source=image)

# Extract detections and their class names
face_detections = face_detections[0].boxes
hand_detections = hand_detections[0].boxes

# Define class names (adjust if necessary)
class_names_face = ['face']
class_names_hand = ['P', 'R', 'S']

# Draw bounding boxes for faces
draw_bounding_boxes(image, face_detections, class_names_face)

# Draw bounding boxes for hands
draw_bounding_boxes(image, hand_detections, class_names_hand)

# Show the image with detections
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the resulting image
output_path = 'output_test_4_image.jpg'  # Change this to your desired output path
cv2.imwrite(output_path, image)
