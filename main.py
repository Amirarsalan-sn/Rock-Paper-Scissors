from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(data='data.yaml', imgsz=640, batch=8, epochs=10, workers=0, device=0)
