from flask import Flask, render_template, url_for, request, send_from_directory
import os
from PIL import Image


app = Flask(__name__)

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
       
    
if __name__ == '__main__':
    app.run(debug=True)
