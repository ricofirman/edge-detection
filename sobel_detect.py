from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO(r"C:\1.ace\edge-detection\best (2)_openvino_model")
cap = cv2.VideoCapture("http://10.205.211.110:4747/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = time.time()
target_size = (800, 600)  # Ukuran target untuk Sobel

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera gagal dibuka")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model.predict(frame, imgsz=320, conf=0.6, verbose=False)

    output = frame.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(output, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(output, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        
        # Scale ke ukuran yang lebih besar
        scale_x = target_size[0] / sobel.shape[1]
        scale_y = target_size[1] / sobel.shape[0]
        scale = min(scale_x, scale_y)
        new_w = int(sobel.shape[1] * scale)
        new_h = int(sobel.shape[0] * scale)
        sobel_besar = cv2.resize(sobel, (new_w, new_h))
        
        # Tambah padding
        padded = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = sobel_besar
        
        cv2.imshow("Sobel ROI (Full Window)", padded)
        break  # Tampilkan hanya ROI pertama

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(output, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("YOLO OpenVINO + Sobel", output)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()