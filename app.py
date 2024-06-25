from flask import Flask, render_template, request, send_file

from ultralytics import YOLO
import cv2
import numpy as np
import os
from sort.sort import *
from util import get_car, read_license_plate, write_csv
from os.path import splitext, basename

app = Flask(__name__, template_folder='templates', static_folder='static')

app.config["IMAGE_UPLOADS"] = "static/images/"

results = {}

mot_tracker = Sort()


# load models
coco_model = YOLO('yolov8n.pt')
model = YOLO('runs/detect/train4/weights/best.pt')

@app.route("/", methods= ["GET"])
def index():
    return render_template('index.html')

@app.route("/", methods= ["POST", "GET"])
def deteksi():
    image = request.files['image']
    img_path = (os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    image.save(img_path)

    img = cv2.imread(img_path)
    img_key = hash(img.tobytes())

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
                    # cv2.imshow('Bbox', loc)
                    text_plate = f'{license_plate_text}'
                    cv2.putText(img, text_plate, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], 'crop.jpg'), license_plate_crop)
        cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], 'gray.jpg'), license_plate_crop_gray)
        cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], 'threshold.jpg'), license_plate_crop_thresh)
        cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], 'aaa.jpg'), eroded_copy)
        cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], 'segmentation.jpg'), image_with_boxes)
        cv2.imwrite(os.path.join(app.config["IMAGE_UPLOADS"], 'contour.jpg'), contour_img)

    return render_template('index.html', img_path = img_path, prediksi = text_plate)

@app.route("/model", methods=['POST', 'GET'])
def train():
    if request.method == "POST":
        try:
            # Get the selected YOLO model from the form
            selected_model = request.form.get("model")

            # Initialize the YOLO model based on the selected model
            yolo_model = YOLO(f"{selected_model}.yaml")

            # Get the uploaded file and form data
            data_file = request.files['data']
            epoch = int(request.form.get("epoch"))

            # Save the uploaded file to a temporary location
            data_file_path = "temp_config.yaml"  # Replace with an appropriate location
            data_file.save(data_file_path)

            # Train the YOLO model
            training_results = yolo_model.train(data=data_file_path, epochs=epoch, resume=True)
            result = "Proses Training Sudah Selesai"

            return render_template('model.html', result=result)

        except Exception as e:
            return render_template('error.html', error=str(e))

    # Render the initial form if it's a GET request
    return render_template('model.html')

@app.route("/data", methods=["GET", "POST"])
def data():
    return render_template('data.html')

@app.route("/detect_folder", methods=["GET", "POST"])
def detect_folder():
    if request.method == "POST":
        
        uploaded_folder = request.files['image']
        text_plate = ""
        text_plate_list = []
        input_file_names = []
        ACC = []
        individual_accuracies = []
        
        # Create a folder to store the results
        results_folder = "static/upload folder"
        os.makedirs(results_folder, exist_ok=True)

        uploaded_folder = request.files.getlist('image')
        for file in uploaded_folder:
            file_path = os.path.join(results_folder, file.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            file.save(file_path)
            nama = splitext(basename(file.filename))[0]
            input_file_names.append(nama)

            img = cv2.imread(file_path)
            img_key = hash(img.tobytes())

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
                                                                            # 'score': license_plate_text_score,
                                                                            'bbox_score': score}}
            
                

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text_plate_list.append(f'{license_plate_text}')
            cv2.putText(img, text_plate, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            acc = calculate_accuracy(nama, license_plate_text)
            ACC.append(acc)

            for input_file_name, plate_info, accuracy in zip(input_file_names, text_plate_list, ACC):
                individual_accuracies.append(accuracy)

            # Calculate the average accuracy
            average_accuracy = sum(individual_accuracies) / len(individual_accuracies)
            average_accuracy = round(average_accuracy, 2)

            # Display the average accuracy
            print(f'Average Accuracy: {average_accuracy}')

        output_file_path = 'output_deteksi.txt'
        with open(output_file_path, 'w') as output_file:
            for input_file_name, plate_info, accuracy in zip(input_file_names, text_plate_list, ACC):
                output_file.write(f'{input_file_name}, {plate_info}, {accuracy} \n')
            output_file.write(f'Akurasi Total: {average_accuracy}')
        
        return send_file(output_file_path, as_attachment=True)
        
    return render_template('detect_folder.html')

def calculate_accuracy(true_labels, predicted_labels):
    print(true_labels)
    print(predicted_labels)

    # Initialize variables to count correct predictions
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total_files = len(true_labels)

    if total_files > 0:
        accuracy = (correct_predictions / total_files) * 100
        print(accuracy)
    else:
        print("Error: total_files is 0")

    return accuracy



app.run(debug=True)