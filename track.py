import cv2
import numpy as np

def track_white_ball(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define a color range for white
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])

        # Create a mask using the color range
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around the white ball
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('White Ball Tracking', frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = 'video.mp4'
track_white_ball(video_path)
