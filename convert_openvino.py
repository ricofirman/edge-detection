from ultralytics import YOLO

model = YOLO(r"C:\1.ace\edge-detection\sobel1.pt")

model.export(
    format="openvino",
    imgsz=320
)