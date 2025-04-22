import time
import os
import asyncio
import cv2
import RPi.GPIO as GPIO
import pytesseract
from PIL import Image
from ultralytics import YOLO
import pyttsx3, pygame, edge_tts

TEXT_BUTTON_PIN = 17
OBJECT_BUTTON_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(TEXT_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(OBJECT_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

IMAGE_PATH = "/home/pi/Desktop/captured.jpg"
VOICE = "en-US-JennyNeural"
OUTPUT_FILE = "AUDIO.mp3"

model = YOLO("yolov5n.pt")

async def s_speak(text):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(OUTPUT_FILE)
    pygame.mixer.init()
    pygame.mixer.music.load(OUTPUT_FILE)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def speak(text):
    asyncio.run(s_speak(text))
def capture_image():
    print("Capturing image from Pi Camera...")
    os.system(f"libcamera-jpeg -o {IMAGE_PATH} --width 1280 --height 720 --nopreview")
    print("Image captured.")

def image_to_text(image_path):
    image = Image.open(image_path)
    rotated_image = image.rotate(-90, expand=True)
    extracted_text = pytesseract.image_to_string(rotated_image)

    if extracted_text.strip():
        print("Extracted Text:\n", extracted_text)
        speak(extracted_text)
    else:
        print("No text detected in the image.")
        speak("No text detected in the image.")

def detect_object():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        results = model.predict(frame, imgsz=320, conf=0.5)
        detected_objects = set()
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                detected_objects.add(label)

        if detected_objects:
            object_list = ", ".join(detected_objects)
            print("Detected objects:", object_list)
            speak(f"Detected objects are: {object_list}")
        else:
            print("No objects detected.")
            speak("No objects detected.")
    else:
        print("Failed to capture image.")
        speak("Failed to capture image.")
print("System is running. Press buttons for respective actions...")

try:
    while True:
        if GPIO.input(TEXT_BUTTON_PIN) == GPIO.LOW:
            print("Text button pressed!")
            capture_image()
            time.sleep(2)
            image_to_text(IMAGE_PATH)
            time.sleep(1)

        if GPIO.input(OBJECT_BUTTON_PIN) == GPIO.LOW:
            print("Object detection button pressed!")
            detect_object()
            time.sleep(1)

except KeyboardInterrupt:
    print("\nProgram stopped. Cleaning up GPIO...")
    GPIO.cleanup() 