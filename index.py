import cv2
import numpy as np
import os

def detect_orange_objects(image, image_id):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 80, 50])  # Adjusted lower bound for darker orange color
    upper_orange = np.array([20, 255, 255])  # Adjusted upper bound for darker orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orange_image = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            if solidity > 0.8:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(orange_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                length = cv2.arcLength(contour, True)
                cv2.putText(orange_image, 'Note', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(f"orange_{image_id}.jpg", orange_image)
    return orange_image

def main():
    folder_path = 'notes'
    image_id = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.png')):
            image = cv2.imread(os.path.join(folder_path, file_name))
            if image is None:
                print(f"Error: Could not read the image {file_name}. Skipping.")
                continue
            image_id += 1
            orange_image = detect_orange_objects(image, image_id)
            print(f"Image {file_name} processed. Orange objects saved as 'orange_{image_id}.jpg'.")

if __name__ == "__main__":
    main()
