import cv2
import numpy as np

def main():
    # Load the SSD-MobileNet model
    model_file = "opencv_face_detector_uint8.pb"
    config_file = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    # Check if the video capture object is opened
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while True:
        # Capture a video frame
        ret, frame = cap.read()

        if not ret:
            break

        # Detect faces in the video frame
        faces = detect_faces(frame, net)

        # Draw bounding boxes around the detected faces
        for (x, y, x2, y2) in faces:
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        # Display the video frame with the face detections
        cv2.imshow("Face Detection", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def detect_faces(frame, net, confidence_threshold=0.6):
    # Preprocess the video frame
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Perform face detection
    net.setInput(blob)
    detections = net.forward()

    # Process the detected faces
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2, y2))

    return faces

if __name__ == "__main__":
    main()