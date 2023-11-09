import time
import cv2
import math
import pytesseract
from gtts import gTTS
import os
from flask import Flask, render_template, request, send_from_directory, jsonify

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.template_folder = os.path.abspath('templates')
app.config['UPLOAD_FOLDER'] = os.path.abspath('uploads')

KNOWN_OBJECT_SIZE = 10  # Size of the known object in the real world (in centimeters)
FOCAL_LENGTH = 1000  # Focal length of the camera (example value, adjust based on your camera)

def calculate_distance(image_width, object_width):
    distance = (KNOWN_OBJECT_SIZE * FOCAL_LENGTH) / object_width
    return distance

@app.route('/')
def splash():
    # Render the splash screen template
    return render_template('splash.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    language = request.form.get('language', 'en')

    # Save the uploaded image
    uploaded_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
    uploaded_file.save(image_path)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (1080, 720))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    canny = cv2.Canny(blur, 100, 200)
    ret, thresh1 = cv2.threshold(
        canny, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=6)
    imgThreshold = cv2.erode(dilation, rect_kernel, iterations=4)

    contours, hierarchy = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    extracted_text = []
   
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]

        text = pytesseract.image_to_string(cropped)
        extracted_text.append(text)

        mytext = text
        # language = 'en'
        obj = gTTS(text=mytext, lang=language, slow=False)
        obj.save("output.mp3")
        os.system("start output.mp3")

        distance = calculate_distance(im2.shape[1], w) / 10  # Divide by 10 to convert to centimeters
        print("Distance to object:", distance, "cm")

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg'), im2)
    processed_image_path = 'processed_image.jpg'

    return jsonify(text=extracted_text, image_path=processed_image_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1)

        if key != -1:
            break

    cap.release()

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg'), frame)
    print("Image captured and saved as captured_image.jpg")

    process_image(os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg'))

    cv2.destroyAllWindows()

@app.route('/capture_realtime', methods=['POST'])
def capture_realtime():
    capture_image()
    return render_template('realtime.html')

if __name__ == '__main__':
    # Delay the execution for 3 seconds
    time.sleep(3)
    # Redirect to the home page
    app.add_url_rule('/', 'home', home)
    app.run()
