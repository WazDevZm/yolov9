import cv2
import torch

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv9 model
model = torch.hub.load('WongKinYiu/yolov9', 'custom', path='weights/yolov9-c.pt')
model.to(device)  # Move model to GPU if available
model.eval()  # Set model to evaluation mode

# Open webcam and set lower resolution for better performance
cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Try increasing the FPS if supported

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (YOLO expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection with reduced input size
    results = model(rgb_frame, size=320)  # Reduce size for faster inference

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
