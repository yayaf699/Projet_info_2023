from flask import Flask, render_template, url_for, request, send_from_directory, send_file
import os
from PIL import Image
import base64
import io
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps


app = Flask(__name__)

model = keras.models.load_model("../model_draw.h5")


class_names = []

with open("../classes.txt", 'r') as f:
    class_names = [line.rstrip('\n') for line in f]

def transform_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    dst = cv2.Canny(gray, 0, 150)
    blured = cv2.blur(dst, (5,5), 0)    
    MIN_CONTOUR_AREA=200
    img_thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    Contours,imgContours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in Contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [X, Y, W, H] = cv2.boundingRect(contour)
    for i in range(20):
        if Y > 0 and X > 0:
            Y=Y-1
            X=X-1
        if Y+H <img.shape[1] and X+H < img.shape[0]:
            H=H+2
            W=W+2
    img = img[Y:Y+H, X:X+W]
    img = Image.fromarray(img)
    img = ImageOps.grayscale(img)
    img = img.resize( (28, 28), Image.Resampling.LANCZOS)
    return np.array(img)

app.config['UPLOAD_FOLDER'] = './image'


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['POST'])

def home():
    
    return render_template('index.html') #image=image_base

@app.route("/save", methods=['POST'])
def save():
    # image_data = request.get_json()['image']
    # img = Image.open(request.get(image_data, stream=True).raw)
    # img.save(os.path.join(app.static_folder, "images", "uploaded_image.png"))
     return "ok"


@app.route('/process_image', methods=['POST'])
def process_image():
    # récupérer les données de l'image base64 envoyées par la requête AJAX
    data = request.form['image']
    # décoder l'image en base64
    img_data = base64.b64decode(data.split(',')[1])
    # ouvrir l'image à l'aide de Pillow
    img = Image.open(io.BytesIO(img_data))
    img = np.array(img)
    img = transform_img(img)
    img = Image.fromarray(img)
    # renvoyer l'image modifiée à la page web
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    img.save('./fichier.jpg')
    return send_file(img_io, mimetype='image/jpeg')
       
    
if __name__ == '__main__':
    app.run(debug=True)
