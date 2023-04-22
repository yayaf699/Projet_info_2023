from flask import Flask, render_template, request, send_file
from PIL import Image
import base64
import io
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps


app = Flask(__name__)

model = keras.models.load_model("./model_draw.h5")


class_names = []

with open("./classes.txt", 'r') as f:
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


def process_image(img_data):
    # décoder l'image en base64
    img_data = base64.b64decode(img_data.split(',')[1])
    # ouvrir l'image à l'aide de Pillow
    img = Image.open(io.BytesIO(img_data))
    img = np.array(img)
    img = transform_img(img)
    img = Image.fromarray(img)
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict_top3():
    # récupérer les données de l'image base64 envoyées par la requête AJAX
    img_data = request.form['image']
    img = process_image(img_data)
    predictions = model.predict(img)
    predictions = predictions[0] 
    
    # Pour recuperer le top 3
    idx = (-predictions).argsort()[:3]
    autre = 1 - predictions[idx[0]] - predictions[idx[1]] - predictions[idx[2]]
    top3 = [predictions[idx[0]],
        predictions[idx[1]],
        predictions[idx[2]],
        autre]
    
    plt.pie(top3, labels=(class_names[idx[0]],
                      class_names[idx[1]],
                      class_names[idx[2]],'others'),
                      autopct=lambda p:'{:.2f}%'.format(p))
    plt.title("Plus haute prédiction donnée : " + class_names[idx[0]])

    # Convertir le diagramme en objet BytesIO
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    plt.close()
    img_io.seek(0)
    

    # Encoder l'image en base64
    img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Envoyer l'image à la page web
    return {'image': img_data}


    

       
    
if __name__ == '__main__':
    app.run(debug=True)
