from flask import Flask, render_template, url_for, request, send_from_directory
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './image'


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['POST'])

def home():
    
    return render_template('index.html') #image=image_base

@app.route("/save", methods=['GET', 'POST'])
def save():
    # if request.method == 'POST':
    #     image = request.files['file']
    #     # print(image)
    #     image.save(os.path.join(app.config['UPLOAD_FOLDER'], "img.jpg"))
    #     print("ok")
    #     return
    # else :
    #     return 
       
    

if __name__ == '__main__':
    app.run(debug=True)
