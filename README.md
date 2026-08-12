# Car Colour & Traffic Detection

## Overview

This project develops a machine learning application for detecting car colours and analyzing traffic images.

The system can:

* Detect cars in an image or webcam feed.
* Predict the colour of detected cars.
* Display a **red rectangle for blue cars**.
* Display **blue rectangles for cars of other colours**.
* Detect and count people present at a traffic signal.
* Provide a graphical user interface with image preview.
* Support both image upload and webcam input.

## Features

* Car detection using YOLO.
* Car colour classification using a trained CNN model.
* Person detection and counting.
* Image upload functionality.
* Real-time webcam detection.
* GUI with input image preview and detection results.

## Project Structure

```text
Car-Colour-and-Traffic-Detection/
│
├── dataset/
├── model/
├── icons/
├── uploads/
├── output/
├── detect.py
├── gui.py
├── requirements.txt
└── README.md
```

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* YOLO
* Ultralytics
* NumPy
* Tkinter
* Pillow

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the GUI:

```bash
python gui.py
```

The application provides options to upload an image and use the webcam.

## Output

The application displays:

* Detected car colour
* Car bounding boxes
* Number of cars detected
* Number of people detected
* Webcam/image preview

## Note

The trained model and datasets used for development are included according to the project submission requirements.
