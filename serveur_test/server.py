from flask import Flask, render_template, url_for, request

app = Flask(__name__)



@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['POST'])
def home():
    image_base= request.form["image"]
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)

