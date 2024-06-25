from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image 
import util
import os
import sys
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
model = YOLO('runs/detect/train4/weights/best.pt')

# load image
img = cv2.imread('data/images/train/AA6862UC.jpg')
img_key = hash(img.tobytes())

# Resize the image to a specific width and height
resized_width, resized_height = 640, 640
img = cv2.resize(img, (resized_width, resized_height))

vehicles = [2, 3, 5, 7]
#mobil, motor, bis, truk

# detect vehicles
detections = coco_model(img)[0]
detections_ = []

for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if int(class_id) in vehicles:
        detections_.append([x1, y1, x2, y2, score])
        
        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = model(img)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 128, 255, cv2.THRESH_BINARY)
                
                total_black_pixels = np.sum(license_plate_crop_thresh == 0)
                total_white_pixels = np.sum(license_plate_crop_thresh == 255)

                if total_black_pixels > total_white_pixels:
                    print('plat hitam')
                    license_plate_crop_thresh = license_plate_crop_thresh
                else:
                    print('plat putih')
                    license_plate_crop_thresh = ~license_plate_crop_thresh
                    
                eroded = license_plate_crop_thresh
                eroded_copy = cv2.cvtColor(eroded.copy(), cv2.COLOR_GRAY2RGB)
                image_with_boxes = license_plate_crop.copy()
                
                hx,wx,cx = image_with_boxes.shape

                contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour_img = np.zeros_like(eroded)

                contours_img = []
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    # if h * hx and w * h:
                    # if h > (1/3 * hx) and h < (1/2 * hx) and w < (1/2 * h + 1/4*h):
                    if h < hx and w < h:
                        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        cv2.rectangle(eroded_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

                        contour_img[y:y+h, x:x+w] = eroded[y:y+h, x:x+w]
                        contours_img.append(contour_img[y:y+h, x:x+w])

                # read license plate
                license_plate_text, license_plate_text_score = read_license_plate(contour_img)
                                
                if img_key not in results:
                    results[img_key] = {}

                if license_plate_text is not None:
                    results[img_key][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,}}
                    
                                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                loc = img.copy()
                cv2.imshow('Bbox', loc)
                text_plate = f'{license_plate_text}'
                cv2.putText(img, text_plate, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the resulting img
cv2.imshow('Images', img)
cv2.imshow('Crop', license_plate_crop)
cv2.imshow('Gray', license_plate_crop_gray)
cv2.imshow('Threshold', license_plate_crop_thresh)
cv2.imshow('aaa', eroded_copy)
cv2.imshow('Segmentasi', image_with_boxes)
cv2.imshow('Contour', contour_img)
print(text_plate)

cv2.waitKey(0)

# Release the capture
cv2.destroyAllWindows()


# # write results
# write_csv(results, './test.csv')