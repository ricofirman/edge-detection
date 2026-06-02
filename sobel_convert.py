import cv2
import numpy as np
import os
from glob import glob

input_folder = r"C:\1.ace\edge-detection\img"
output_folder = r"C:\1.ace\edge-detection\img_sobel"

os.makedirs(output_folder, exist_ok=True)

image_files = glob(os.path.join(input_folder, "*.png"))

for img_path in image_files:

    img = cv2.imread(img_path)

    if img is None:
        print(f"Gagal membaca: {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    sobel = cv2.magnitude(sobelx, sobely)

    sobel = np.uint8(np.clip(sobel, 0, 255))

    filename = os.path.basename(img_path)

    save_path = os.path.join(
        output_folder,
        filename
    )

    cv2.imwrite(save_path, sobel)

    print(f"Berhasil: {filename}")

print("Selesai konversi semua gambar ke Sobel")