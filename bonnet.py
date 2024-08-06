import cv2
import os
import supervision as sv
from ultralytics import YOLOv10
import typer

# Load the YOLOv10 model
model = YOLOv10("best.pt")

# Create a Typer app for command-line interfaces
app = typer.Typer()

# Dictionary to map category IDs to names
category_dict = {
    0: 'bonnet'  # ID 0 corresponds to 'bonnet'
}

def process_webcam():
    # Open the default webcam (usually 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set up annotators for bounding boxes and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Use the model to detect objects in the frame
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Draw bounding boxes and labels for detected objects
        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict[class_id]  # Get the name of the class from the ID
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            # Add the class name and confidence score to the frame
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the frame with annotations
        cv2.imshow("Webcam", frame)
        # Exit the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the Typer app if this script is executed directly
if __name__ == "__main__":
    app()
