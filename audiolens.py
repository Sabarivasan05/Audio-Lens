import pytesseract, os, playsound, cv2
import pyttsx3
import RPi.GPIO as GPIO
from ultralytics import YOLO
import time


model = YOLO('yolov8n.pt')

BUTTON_TEXT = 17
BUTTON_OBJECT = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_TEXT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_OBJECT, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


def capture_image_for_text():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    path = "CAPTURED_IMAGE.png"
    cv2.imwrite(path, frame)
    camera.release()
    cv2.destroyAllWindows()
    extract_text_from_image(path)

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray_image)
    text_to_audio(extracted_text)

def text_to_audio(text):
    print("Extracted Text:\n", text.strip())
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    audio_file = "audio.mp4"
    engine.save_to_file(text, audio_file)
    engine.runAndWait()
    playsound.playsound(audio_file)

    if os.path.exists("CAPTURED_IMAGE.png"):
        os.remove("CAPTURED_IMAGE.png")
    if os.path.exists(audio_file):
        os.remove(audio_file)


def cam_for_object_detection():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    path = 'o_d_img.png'
    cv2.imwrite(path, frame)
    cam.release()
    cv2.destroyAllWindows()
    print('Image captured for object detection')
    extract_object_from_image(path)

def extract_object_from_image(path):
    results = model(path)
    detected_objects = results[0].names
    detected_labels = results[0].boxes.cls
    object_names = [detected_objects[int(label)] for label in detected_labels]
    if object_names:
        detected = ", ".join(object_names)
        print("Detected objects:", detected)
        text_to_audio("Detected objects are " + detected)
    else:
        print("No objects detected.")
        text_to_audio("No objects detected.")
    os.remove(path)


try:
    print("Waiting for button press...")
    while True:
        if GPIO.input(BUTTON_TEXT) == GPIO.LOW:
            print("Text extraction button pressed.")
            capture_image_for_text()
            time.sleep(0.5)  # Debounce

        if GPIO.input(BUTTON_OBJECT) == GPIO.LOW:
            print("Object detection button pressed.")
            cam_for_object_detection()
            time.sleep(0.5)  # Debounce

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    GPIO.cleanup()
