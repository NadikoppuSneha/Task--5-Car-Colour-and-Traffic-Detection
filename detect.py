import os
import json
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# =========================
# PATHS
# =========================
MODEL_PATH = "model/car_colour_model.keras"
CLASSES_PATH = "model/classes.json"

# =========================
# LOAD MODELS
# =========================
print("Loading Car Colour Model...")
colour_model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

print("Car Colour Model Loaded Successfully!")
print("Colours:", list(classes.values()))

print("Loading YOLO model...")
detector = YOLO("yolov8n.pt")
print("YOLO Model Loaded Successfully!")

# COCO classes
CAR_CLASS = 2
PERSON_CLASS = 0

# =========================
# COLOUR PREDICTION
# =========================
def predict_colour(car_crop):
    if car_crop is None or car_crop.size == 0:
        return "unknown", 0.0

    image = cv2.resize(car_crop, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = colour_model.predict(image, verbose=0)[0]

    index = int(np.argmax(prediction))
    confidence = float(prediction[index])

    colour = classes[str(index)]

    return colour, confidence


# =========================
# DRAW DETECTION
# =========================
def process_frame(frame):
    results = detector(frame, verbose=False)

    car_count = 0
    person_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence < 0.40:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # PERSON
            if cls == PERSON_CLASS:
                person_count += 1

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Person",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # CAR
            elif cls == CAR_CLASS:
                car_count += 1

                car_crop = frame[
                    max(0, y1):min(frame.shape[0], y2),
                    max(0, x1):min(frame.shape[1], x2)
                ]

                colour, colour_conf = predict_colour(car_crop)

                # REQUIRED TASK:
                # BLUE CAR -> RED rectangle
                # OTHER COLOURS -> BLUE rectangle
                if colour.lower() == "blue":
                    box_colour = (0, 0, 255)       # RED
                else:
                    box_colour = (255, 0, 0)       # BLUE

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_colour,
                    3
                )

                label = f"{colour} {colour_conf * 100:.1f}%"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_colour,
                    2
                )

    # COUNTS
    cv2.rectangle(frame, (10, 10), (300, 85), (30, 30, 30), -1)

    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return frame


# =========================
# WEBCAM
# =========================
print("\nStarting Traffic Detection...")
print("Press Q to quit.")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam Started!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    frame = process_frame(frame)

    cv2.imshow(
        "Car Colour & Traffic Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection stopped.")