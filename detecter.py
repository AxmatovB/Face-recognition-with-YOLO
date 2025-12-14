import cv2
from ultralytics import YOLO
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

model = YOLO("/home/axmatov/Desktop/AI/runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(0)
save_dir = "natija"
os.makedirs(save_dir, exist_ok=True)

saved_once = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            if label.lower() == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                if not saved_once:
                    cropped = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                    canny_edges = cv2.Canny(blurred, 30, 100)
                    cv2.imwrite(os.path.join(save_dir, "canny.jpg"), canny_edges)

                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    sobel_edges = cv2.magnitude(sobelx, sobely)
                    sobel_edges = np.uint8(sobel_edges)
                    cv2.imwrite(os.path.join(save_dir, "sobel.jpg"), sobel_edges)

                    similarity, _ = ssim(canny_edges, sobel_edges, full=True)
                    similarity_percent = similarity * 100

                    comparison = np.hstack((canny_edges, sobel_edges))
                    comparison_color = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)

                    text = f"O‘xshashlik: {similarity_percent:.2f}%"
                    cv2.putText(comparison_color, text, (10, comparison_color.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    cv2.imwrite(os.path.join(save_dir, "comparison.jpg"), comparison_color)
                    print("Natijalar saqlandi: natija/canny.jpg, natija/sobel.jpg, natija/comparison.jpg")

                    saved_once = True  
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

