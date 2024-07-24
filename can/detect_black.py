import cv2
import numpy as np
import threading

class LineDetector:
    def __init__(self, ip):
        self.ip = ip
        self.status = 'NULL'

    def detect_line(self):
        # Open the camera
        cap = cv2.VideoCapture(self.ip)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        while True:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("Cannot get frame")
                break

            # Detect right-angled corners formed by black lines
            processed_frame, status = self.detect_black_right_angle_corners(frame)
            self.status = status

            cv2.imshow('Camera', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Get the current status
        current_status = self.get_status()
        print(f"Current Status: {current_status}")

        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def detect_black_right_angle_corners(self, frame):
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

        return resized_image, "not turn"

    def get_status(self):
        return self.status


# Usage example
def main():
    left_camera_ip = 2  # Replace with your camera IP or camera index
    detector = LineDetector(left_camera_ip)

    # Create a thread and start detection
    thread1 = threading.Thread(target=detector.detect_line)
    thread1.start()

    # Wait for the thread to finish
    thread1.join()

if __name__ == "__main__":
    main()
