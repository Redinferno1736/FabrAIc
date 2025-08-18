import os
import io
from flask import Flask, redirect, url_for, session, request, render_template, flash, send_file, url_for
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
from pymongo.errors import DuplicateKeyError
from bson.objectid import ObjectId
from datetime import datetime, timezone
from bson import ObjectId
from gridfs import GridFSBucket

load_dotenv() 

app = Flask(__name__)
oauth = OAuth(app)

client = MongoClient(os.getenv("MONGO_CLIENT"))
db = client['userinfo']
collection = db['users']

app.secret_key = os.getenv("SECRET_KEY")  

google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    userinfo_endpoint="https://www.googleapis.com/oauth2/v3/userinfo",
    client_kwargs={"scope": "openid email profile"},
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login",methods=["GET","POST"])
def login_page():
    if request.method=="POST":
        username=request.form.get("username")
        password=request.form.get("password")
        try:
            user = collection.find_one({'username': username})
            if check_password_hash(user["hash"], password): 
                session['username'] = username
                return redirect('/home')
            else:
                flash("Invalid username or password!")
                return render_template("login.html")
        except Exception as e:
            flash("An unexpected error occurred: " + str(e))
            return render_template("login.html")
    else:
        return render_template("login.html")

@app.route("/register",methods=["GET","POST"])
def register_page():
    if request.method=="POST":
        username=request.form.get("username")
        password=request.form.get("password")
        confirm=request.form.get("confirm")
        print(username,password,confirm)
        if password != confirm:
            flash("Passwords do not match!")
            return render_template("register.html")
        hash = generate_password_hash(password)
        try:
            collection.insert_one({'username': username, 'hash': hash})
            return redirect('/login')  
        except DuplicateKeyError:
            flash("Username has already been registered!")
            return render_template("register.html")
        except Exception as e:
            flash("An unexpected error occurred: " + str(e))
            return render_template("register.html")
    else:
        return render_template("register.html")
    
@app.route("/auth/google")
def google_login():
    redirect_uri = url_for("google_callback", _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route("/auth/google/callback")
def google_callback():
    token = google.authorize_access_token()
    user_info = google.get("https://www.googleapis.com/oauth2/v2/userinfo").json()
    session["user"] = user_info
    user = collection.find_one({'username': session["user"]['email']})
    if not user:
        collection.insert_one({'username': session["user"]['email'],'name':session["user"]['name']})
    session['username'] = session["user"]['email']
    return redirect('/home') 

@app.route("/home", methods=["GET", "POST"])
def home():
    username = session.get('username')
    if username:
        return render_template("home.html")
    return redirect('/login')

@app.route("/logout")
def logout():
    session.clear()
    return redirect('/') 

if __name__ == "__main__":
    app.run(debug=True)
