from ultralytics import YOLO
import cv2
import numpy as np
import time

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(r"C:\1.ace\edge-detection\sobel1.pt")

# ==========================
# CAMERA
# ==========================
cap = cv2.VideoCapture("http://10.205.211.110:4747/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera gagal dibuka")
        break

    frame = cv2.resize(frame, (640, 480))

    # ==========================
    # ORIGINAL
    # ==========================
    original = frame.copy()

    # ==========================
    # SOBEL (SAMA SEPERTI TRAINING)
    # ==========================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    sobelx = cv2.Sobel(
        blur,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobely = cv2.Sobel(
        blur,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel = cv2.magnitude(
        sobelx,
        sobely
    )

    sobel = np.uint8(
        np.clip(
            sobel,
            0,
            255
        )
    )

    # YOLO perlu 3 channel
    sobel_bgr = cv2.cvtColor(
        sobel,
        cv2.COLOR_GRAY2BGR
    )

    # ==========================
    # YOLO DETECT
    # ==========================

    results = model.predict(
        source=sobel_bgr,
        imgsz=320,
        conf=0.6,
        verbose=False
    )

    yolo_view = sobel_bgr.copy()

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        conf = float(
            box.conf[0]
        )

        cls = int(
            box.cls[0]
        )

        label = model.names[cls]

        cv2.rectangle(
            yolo_view,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            yolo_view,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # ==========================
    # FPS
    # ==========================

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.0001
    )

    prev_time = current_time

    cv2.putText(
        original,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        yolo_view,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ==========================
    # DISPLAY
    # ==========================

    cv2.imshow(
        "1 - Original Camera",
        original
    )

    cv2.imshow(
        "2 - Sobel",
        sobel
    )

    cv2.imshow(
        "3 - YOLO Detection",
        yolo_view
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()