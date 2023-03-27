from flask import Flask, render_template, url_for, Request
import base64 
import os


app = Flask(__name__)



@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['POST'])

def home():
    
    return render_template('index.html' )#image=image_base

@app.route("/save", methods=["POST"])
def save():
    image = Request.form["form"]

    image = base64.b64decode(image.split(",")[1])


    with open(os.path.join("/image", "image.png"), "wb") as f:# write binary mode
        f.write(image)

    return "ok"

if __name__ == '__main__':
    app.run(debug=True)
