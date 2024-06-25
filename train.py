from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
result = model.train(data="config.yaml", epochs=1, resume=True, visualize=True)  # train the model
