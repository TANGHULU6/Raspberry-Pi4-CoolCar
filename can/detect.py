import socket
import threading
import time
import cv2
import numpy as np

from detect_black import LineDetector
from ultralytics import YOLO, YOLOWorld

control_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
control_client_socket.connect(('172.25.107.151',8007))  # Raspberry Pi's IP address
model = YOLOWorld('yolov8s-world.pt')

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def if_stop(red_x, red_y):
    print(red_x)
    print(red_y)
    if red_y > 180 and red_y < 270:
        if 100 < red_x < 300:
            return True
        else:
            return False
    else:
        return False
    
def if_stop2(red_x2, red_y2):
    print(red_x2)
    print(red_y2)
    if red_y2 > 270 and red_y2 < 370:
        if 380 < red_x2 < 390:
            return True
        else:
            return False
    else:
        return False
    
def if_stop3(red_x3, red_y3):
    print(red_x3)
    print(red_y3)
    if red_y3 > 270 and red_y3 < 370:
        if 490 < red_x3 < 500:
            return True
        else:
            return False
    else:
        return False

def detect_hutao(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检查图像的区域是否为低对比度区域（白纸区域）
    central_region = gray[217:270, 279:333]
    std_dev = np.std(central_region)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(std_dev)
    if std_dev < 10:  # 假设低对比度（白纸区域）的阈值
        return False
    else:
        return True

def detect_hutao1(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检查图像的区域是否为低对比度区域（白纸区域）
    central_region = gray[215:270, 378:419]
    std_dev = np.std(central_region)
    print(std_dev)
    if std_dev < 10:  # 假设低对比度（白纸区域）的阈值
        return False
    else:
        return True
    
def detect_hutao2(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检查图像的区域是否为低对比度区域（白纸区域）
    central_region = gray[215:267, 479:515]
    std_dev = np.std(central_region)
    print(std_dev)
    if std_dev < 10:  # 假设低对比度（白纸区域）的阈值
        return False
    else:
        return True

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
    return processed_frame, "turn"


# Start capturing video from the camera
cap = cv2.VideoCapture(2)
preturn = False
turning = False
stop = False
enable_stop = False
cnt = 0
num = 2
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


    # Define the range for the blue color
    lower_blue = np.array([80, 100, 0])
    upper_blue = np.array([140, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours of the red areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_red_x = 1000000
    center_red_y = 1000000
    diagonal_angle = 0
    aligned = False
    # Draw bounding boxes around detected red areas and output center coordinates and area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50 and area < 500:  # Adjusted area range
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate center coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Print center coordinates and area
            # print(f"Center coordinates: ({center_x}, {center_y}), Area: {area}")
            # print(f"h:{h}, w:{w}")
            if w/h >= 5.5:
                print("Aligned!")
                aligned = True
            diagonal_angle = np.degrees(np.arctan2(h, w))
            if h/w >= 4 and enable_stop:
                stop = True
            center_red_x = center_x
            center_red_y = center_y

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_blue_x = 1000000
    center_blue_y = 1000000
    # Draw bounding boxes around detected blue areas and output center coordinates and area
    for contour in contours_blue:
            area = cv2.contourArea(contour)
            if area > 100 and area < 800:  # Adjusted area range
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Calculate center coordinates
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Print center coordinates and area
                print(f"Center coordinates: ({center_x}, {center_y}), Area: {area}")
                center_blue_x = center_x
                center_blue_y = center_y

    # print(center_red_y)
    # print(center_blue_y)
    if stop:
        cnt += 1
        mess = "S"
        control_client_socket.sendall(mess.encode())
        time.sleep(0.2)
        if cnt > 25:
            control_client_socket.sendall("L".encode())
            time.sleep(0.2)
            control_client_socket.sendall("W".encode())
            combined_frame = np.hstack((processed_frame, frame))
            cv2.imshow('Combined Detection', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(10)
            out = 0
            while out < 35:
                out += 1
                messw = "W"
                control_client_socket.sendall(messw.encode())
                time.sleep(0.2)
            
            preturn = False
            enable_stop = False
            cnt = 0
            
        else:
            combined_frame = np.hstack((processed_frame, frame))
            cv2.imshow('Combined Detection', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    message = ''
    if center_blue_x < center_red_x and center_blue_y > center_red_y:
        print(f"R{diagonal_angle}")
        message = "D"
    elif center_blue_x > center_red_x and center_blue_y > center_red_y:
        print(f"R{90 + diagonal_angle}")
        message = "D"
    elif center_blue_x < center_red_x and center_blue_y < center_red_y:
        print(f"L-{diagonal_angle}")
        message = "A"
    elif center_blue_x > center_red_x and center_blue_y < center_red_y:
        print(f"L-{90 + diagonal_angle}")
        message = "A"
    else:
        print("same xy")
    time.sleep(0.1)
    if (diagonal_angle > 10 or not aligned) and not preturn:
        preturn = False
        turning = False
        stop = False
        control_client_socket.sendall(message.encode())
        # time.sleep(1)
    else:
        preturn = True
        messageTurn = ''
        if (if_stop(center_red_x, center_red_y) or turning):
            turning = True
            enable_stop = True
            messageTurn = 'A'
        elif (if_stop2(center_red_x, center_red_y) or turning) and not detect_hutao1(frame):
            turning = True
            enable_stop = True
            messageTurn = 'D'
        elif (if_stop3(center_red_x, center_red_y) or turning) and not detect_hutao2(frame):
            turning = True
            enable_stop = True
            messageTurn = 'D'
        else:
            messageTurn = 'W'
        control_client_socket.sendall(messageTurn.encode())
        # time.sleep(1)
        
    # Display the resulting frame
    model.set_classes(["person", "laptop"])
    a = model.predict(source=frame, save=False)
    combined_frame = np.hstack((processed_frame, frame, a[0].plot()))
    # combined_frame = np.hstack((processed_frame, frame))

    # Display the resulting frame
    cv2.imshow('Combined Detection', combined_frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()





