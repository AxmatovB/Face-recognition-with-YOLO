# Real-Time Person Detection and Edge Comparison

This project detects a **person** from a webcam using a trained **YOLO model**, then applies **Canny** and **Sobel** edge detection on the detected area and compares the results using **SSIM (Structural Similarity Index)**.

The project is intended for **learning, experimentation, and computer vision practice**.

---

## What this project does

* Opens the webcam
* Detects a **person** in real time
* Draws a bounding box around the detected person
* Crops the detected person area
* Applies **Canny edge detection**
* Applies **Sobel edge detection**
* Compares both edge results using **SSIM**
* Saves all results as image files

---

## How it works

1. The webcam starts and captures frames
2. Each frame is processed by the YOLO model
3. When a **person** is detected:

   * A bounding box is drawn
   * The detected area is cropped
4. The cropped image is converted to grayscale
5. Canny and Sobel edge detection are applied
6. Both results are compared using SSIM
7. Output images are saved to disk (only once)

---

## Project structure

```
.
├── detecter.py        # Main Python script
├── data.yaml          # Dataset configuration file
├── runs/              # YOLO training results
│   └── detect/train2/weights/best.pt
├── natija/            # Output folder (auto-created)
│   ├── canny.jpg
│   ├── sobel.jpg
│   └── comparison.jpg
└── README.md
```

---

## Requirements

Install required libraries:

```bash
pip install ultralytics opencv-python numpy scikit-image
```

---

## How to run

1. Set the correct path to the YOLO model in `detecter.py`:

```python
model = YOLO("path/to/best.pt")
```

2. Run the script:

```bash
python detecter.py
```

3. Press **Q** to stop the program.

---

## Output

* `canny.jpg` – Result of Canny edge detection
* `sobel.jpg` – Result of Sobel edge detection
* `comparison.jpg` – Side-by-side comparison with SSIM value

---

## Notes

* Detection works only for the **person** class
* Uses the default webcam (`VideoCapture(0)`)
* Results are saved only once
* GPU is optional but improves performance

---

## Technologies used

* Python
* YOLO (Ultralytics)
* OpenCV
* NumPy
* scikit-image (SSIM)

---

## License

This project is provided for **educational and experimental purposes**.
