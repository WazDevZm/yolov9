import cv2
import torch

# Load the YOLOv9 model
model = torch.hub.load('WongKinYiu/yolov9', 'custom', path='weights/yolov9-c.pt')

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('YOLOv9 Real-Time Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
