from ultralytics import YOLO

model = YOLO(r"C:\1.ace\edge-detection\best (1).pt")

model.export(
    format="openvino",
    imgsz=320
)