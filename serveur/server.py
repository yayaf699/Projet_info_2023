from flask import Flask, render_template, url_for
from flask import RegistrationForm, LoginForm
app = Flask(__name__)

app.config['SECRET_KEY'] = '0aa3b2e238ca0390b8bebd82dd583841'

posts = [
    {
        'author' :'Sofiane Chogli', 
        'title' : 'Blog Post 1',
        'content' : 'First Post Content', 
        'date_posted' : '6 Mars, 2023'

    },
    {
        'author' :'Sofiane Chogli', 
        'title' : 'Blog Post 2',
        'content' : 'Second Post Content', 
        'date_posted' : '7 Mars, 2023'

    
    }
]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)

@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register")
def register():
    form = RegistrationForm()
    return render_template('register.html', title='Register', form = form)


@app.route("/login")
def login():
    form = LoginForm()
    return render_template('login.html', title='Login', form = form)


if __name__ == '__main__':
    app.run(debug=True)
