import threading
import cv2
import numpy as np

from detect_black import LineDetector

def detect_black_right_angle_corners(frame):
    # Resize the image to 25% of its original size
    scale_percent = 25
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect right-angled corners
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Check if the contour is a rectangle (four sides)
        if len(approx) == 4:
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Calculate angles between edges
            angles = []
            for i in range(4):
                j = (i + 1) % 4
                k = (i + 2) % 4
                vector1 = approx[j] - approx[i]
                vector2 = approx[k] - approx[j]
                dot_product = np.dot(vector1.flatten(), vector2.flatten())
                magnitude1 = np.linalg.norm(vector1)
                magnitude2 = np.linalg.norm(vector2)
                cosine_angle = dot_product / (magnitude1 * magnitude2)
                angle = np.arccos(cosine_angle) * 180 / np.pi
                angles.append(angle)

            # Check if any angle is approximately 90 degrees (adjust threshold as needed)
            if all(abs(angle - 90) < 15 for angle in angles):  # Adjust tolerance as needed
                # Calculate area of the rectangle
                area = w * h

                # Output if the area is within the specified range
                if 800 <= area <= 1000:
                    # Draw the rectangle
                    cv2.drawContours(resized_image, [approx], -1, (0, 255, 0), 2)

                    # Calculate center point coordinates
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Output center coordinates and area
                    print(f"Center coordinates: ({center_x}, {center_y}), Area: {area}")
    # Resize back to original size
    processed_frame = cv2.resize(resized_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    return processed_frame, "not turn"

# Start capturing video from the camera
cap = cv2.VideoCapture(2)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    
    # Detect right-angled corners formed by black lines
    processed_frame, status = detect_black_right_angle_corners(frame)

    # cv2.imshow('Camera', processed_frame)
    print(status)

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the red color
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of the red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected red areas and output center coordinates and area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 and area < 500:  # Adjusted area range
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate center coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Print center coordinates and area
            print(f"Center coordinates: ({center_x}, {center_y}), Area: {area}")

    # Display the resulting frame
    # cv2.imshow('Red Part Detected', frame)
    combined_frame = np.hstack((processed_frame, frame))

    # Display the resulting frame
    cv2.imshow('Combined Detection', combined_frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
