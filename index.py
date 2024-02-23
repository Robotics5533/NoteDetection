import cv2
import numpy as np
import os

def get_dominant_orange_hue(image, x, y, w, h):
    roi = image[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 80, 50]) 
    upper_orange = np.array([20, 255, 255])
    mask_orange = cv2.inRange(hsv_roi, lower_orange, upper_orange)
    hist_orange = cv2.calcHist([hsv_roi], [0], mask_orange, [180], [0, 180])
    max_idx_orange = np.argmax(hist_orange)
    dominant_orange_hue = max_idx_orange * 2
    return dominant_orange_hue


def detect_orange_objects(image, image_id):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 80, 50])
    upper_orange = np.array([20, 255, 255])
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
                dominant_hue = get_dominant_orange_hue(image, x, y, w, h)
                if dominant_hue >= 5 and dominant_hue <= 30:
                    cv2.rectangle(orange_image, (x, y), (x + w, y + h), (128, 0, 128), 2)
                    cv2.putText(orange_image, 'Note', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1, cv2.LINE_AA)  # Purple color
    cv2.imwrite(f"orange_{image_id}.jpg", orange_image)
    return orange_image

def main():
    folder_path = 'notes'
    output_folder = 'results'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_id = 0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.png')):
            image = cv2.imread(os.path.join(folder_path, file_name))
            if image is None:
                print(f"Error: Could not read the image {file_name}. Skipping.")
                continue
            image_id += 1
            processed_image = detect_orange_objects(image, image_id)
            output_path = os.path.join(output_folder, f"orange_{image_id}.jpg")
            cv2.imwrite(output_path, processed_image)
            print(f"Image {file_name} processed. Orange objects saved as '{output_path}'.")



if __name__ == "__main__":
    main()
