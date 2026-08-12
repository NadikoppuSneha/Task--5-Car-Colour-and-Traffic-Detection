import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import tensorflow as tf
from ultralytics import YOLO


# =========================
# LOAD MODELS
# =========================

print("Loading models...")

colour_model = tf.keras.models.load_model(
    "model/car_colour_model.keras"
)

with open("model/classes.json", "r") as f:
    classes = json.load(f)

detector = YOLO("yolov8n.pt")

print("Models loaded successfully!")


# =========================
# COLOUR PREDICTION
# =========================

def predict_colour(car_crop):

    if car_crop is None or car_crop.size == 0:
        return "unknown", 0.0

    image = cv2.resize(car_crop, (128, 128))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    prediction = colour_model.predict(
        image,
        verbose=0
    )[0]

    index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[index]
    )

    return classes[str(index)], confidence


# =========================
# PROCESS IMAGE
# =========================

def process_image(image):

    results = detector(
        image,
        verbose=False
    )

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

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # =====================
            # PERSON
            # =====================

            if cls == 0:

                person_count += 1

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    "Person",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # =====================
            # CAR
            # =====================

            elif cls == 2:

                car_count += 1

                car_crop = image[
                    max(0, y1):min(image.shape[0], y2),
                    max(0, x1):min(image.shape[1], x2)
                ]

                colour, colour_conf = predict_colour(
                    car_crop
                )

                # BLUE CAR = RED RECTANGLE
                # OTHER CAR = BLUE RECTANGLE

                if colour.lower() == "blue":
                    box_colour = (0, 0, 255)
                else:
                    box_colour = (255, 0, 0)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    box_colour,
                    3
                )

                label = (
                    f"{colour} "
                    f"{colour_conf * 100:.1f}%"
                )

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    box_colour,
                    2
                )

    return image, car_count, person_count


# =========================
# UPLOAD IMAGE
# =========================

def upload_image():

    path = filedialog.askopenfilename(
        title="Select Traffic Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    if not path:
        return

    image = cv2.imread(path)

    if image is None:

        messagebox.showerror(
            "Error",
            "Could not open image."
        )

        return

    result, cars, people = process_image(
        image
    )

    show_result(
        result,
        cars,
        people
    )


# =========================
# SHOW RESULT
# =========================

def show_result(
    image,
    cars,
    people
):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        image
    )

    image.thumbnail(
        (800, 450)
    )

    photo = ImageTk.PhotoImage(
        image
    )

    image_label.config(
        image=photo
    )

    image_label.image = photo

    car_label.config(
        text=f"Cars Detected: {cars}"
    )

    people_label.config(
        text=f"People Detected: {people}"
    )


# =========================
# WEBCAM
# =========================

cap = None
camera_running = False


def start_camera():

    global cap
    global camera_running

    if camera_running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        messagebox.showerror(
            "Camera Error",
            "Could not open webcam."
        )

        return

    camera_running = True

    update_camera()


def update_camera():

    global cap
    global camera_running

    if not camera_running:
        return

    ret, frame = cap.read()

    if not ret:

        stop_camera()
        return

    result, cars, people = process_image(
        frame
    )

    show_result(
        result,
        cars,
        people
    )

    root.after(
        30,
        update_camera
    )


def stop_camera():

    global cap
    global camera_running

    camera_running = False

    if cap is not None:

        cap.release()
        cap = None


# =========================
# GUI
# =========================

root = tk.Tk()

root.title(
    "Car Colour & Traffic Detection"
)

root.geometry(
    "950x700"
)

root.configure(
    bg="#202124"
)


# TITLE

title = tk.Label(
    root,
    text="CAR COLOUR & TRAFFIC DETECTION",
    font=("Arial", 22, "bold"),
    bg="#202124",
    fg="white"
)

title.pack(
    pady=15
)


# IMAGE PREVIEW

image_label = tk.Label(
    root,
    text="Traffic image preview",
    font=("Arial", 14),
    bg="#303134",
    fg="white",
    width=80,
    height=25
)

image_label.pack(
    pady=10
)


# BUTTON FRAME

button_frame = tk.Frame(
    root,
    bg="#202124"
)

button_frame.pack(
    pady=10
)


# UPLOAD BUTTON

upload_button = tk.Button(
    button_frame,
    text="UPLOAD IMAGE",
    command=upload_image,
    width=20,
    height=2
)

upload_button.grid(
    row=0,
    column=0,
    padx=10
)


# WEBCAM BUTTON

start_button = tk.Button(
    button_frame,
    text="START WEBCAM",
    command=start_camera,
    width=20,
    height=2
)

start_button.grid(
    row=0,
    column=1,
    padx=10
)


# STOP BUTTON

stop_button = tk.Button(
    button_frame,
    text="STOP WEBCAM",
    command=stop_camera,
    width=20,
    height=2
)

stop_button.grid(
    row=0,
    column=2,
    padx=10
)


# CAR COUNT

car_label = tk.Label(
    root,
    text="Cars Detected: 0",
    font=("Arial", 15, "bold"),
    bg="#202124",
    fg="white"
)

car_label.pack(
    pady=5
)


# PEOPLE COUNT

people_label = tk.Label(
    root,
    text="People Detected: 0",
    font=("Arial", 15, "bold"),
    bg="#202124",
    fg="white"
)

people_label.pack(
    pady=5
)


# CLOSE WINDOW

def close_app():

    stop_camera()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


# START GUI

print("OPENING GUI WINDOW...")

root.mainloop()