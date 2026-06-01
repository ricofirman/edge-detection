from ultralytics import YOLO
import cv2

# Load model YOLOv8
model = YOLO(r"C:\1.ace\edge-detection\best.pt")

# URL DroidCam
url = "http://10.205.211.154:4747/video"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Tidak bisa terhubung ke DroidCam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame gagal dibaca")
        break

    # Prediksi YOLO
    results = model(frame, conf=0.25)

    # Gambar bounding box
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Steel Defect Detection", annotated_frame)

    key = cv2.waitKey(1)

    # ESC untuk keluar
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()