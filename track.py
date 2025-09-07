from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
# yolov8m.pt is a medium-sized model, offering a balance between accuracy and speed.
model = YOLO('yolov8m.pt')

# Open the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps # time difference between frames

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracked_video.mp4', fourcc, fps, (width, height))

# Store the track history
# defaultdict is used to easily append to a list for a new track ID
track_history = defaultdict(lambda: [])

# Pixel to meter conversion
# This is a critical parameter for accurate speed calculation.
# The current implementation attempts to find a table and assumes its width is 1.525m.
# This is a very rough estimation and will likely be inaccurate.
# For accurate results, a proper calibration of the camera is needed.
PIXELS_PER_METER = None

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # We set a low confidence threshold to detect the ball even if the model is not very confident
        # We are using the bytetrack tracker, which is more robust to occlusions.
        results = model.track(frame, persist=True, classes=[32], conf=0.1, tracker='bytetrack.yaml') 

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu() if results[0].boxes is not None else []
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes is not None and results[0].boxes.id is not None else []

        # --- Table detection for pixel to meter conversion ---
        if PIXELS_PER_METER is None:
            # This table detection is very basic and likely to fail.
            # It looks for a 'dining table' which is not ideal for a table tennis table.
            # A more robust solution would be to train a specific model for table tennis tables
            # or to manually define the table area.
            table_results = model(frame)
            for r in table_results:
                for c in r.boxes.cls:
                    if model.names[int(c)] == 'dining table':
                        table_box = r.boxes.xyxy[0].cpu().numpy()
                        table_width_pixels = table_box[2] - table_box[0]
                        PIXELS_PER_METER = table_width_pixels / 1.525 # assume table width is 1.525m
                        print(f"Table detected. PIXELS_PER_METER set to: {PIXELS_PER_METER}")
                        break
            if PIXELS_PER_METER is None:
                # If no table is detected, use a default value and warn the user
                PIXELS_PER_METER = 500 #  Fallback value, this needs to be adjusted
                print("Warning: Table not detected. Using default PIXELS_PER_METER. Speed calculation will be inaccurate.")


        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks and calculate speed
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 2:
                # Calculate speed
                dx = track[-1][0] - track[-2][0]
                dy = track[-1][1] - track[-2][1]
                distance_pixels = np.sqrt(dx**2 + dy**2)
                speed_mps = distance_pixels / (PIXELS_PER_METER * dt)
                speed_kmh = speed_mps * 3.6

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                # Display the speed
                cv2.putText(annotated_frame, f"Speed: {speed_kmh:.2f} km/h", (int(x), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
