import os
import io
import pickle
import numpy as np
import base64
from flask import Flask, redirect, url_for, session, request, render_template, flash, send_file, jsonify
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
from pymongo.errors import DuplicateKeyError
from bson.objectid import ObjectId
from datetime import datetime
from gridfs import GridFSBucket
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv() 

class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories=50, num_attributes=25):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet50(weights='DEFAULT') 
        n_inputs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.category_head = nn.Linear(n_inputs, num_categories)
        self.attribute_head = nn.Linear(n_inputs, num_attributes)

    def forward(self, x):
        features = self.backbone(x)
        cat_output = self.category_head(features)
        attr_output = self.attribute_head(features)
        return cat_output, attr_output

class FashionSearchEngine:
    def __init__(self, model_path, index_path, device='cpu'):
        self.device = torch.device(device)
        print(f"Loading model from {model_path}...")
        
        self.model = MultiTaskResNet(num_categories=50, num_attributes=25)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")
            
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading index from {index_path}...")
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
                self.dataset_paths = data["paths"]
                self.dataset_embeddings = data["embeddings"]
        except Exception as e:
            print(f"Error loading index: {e}")
            self.dataset_paths = []
            self.dataset_embeddings = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def get_query_embedding(self, image_stream):
        img = Image.open(image_stream).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model.backbone(img_tensor)
        return feature.cpu().numpy()

    def search(self, image_stream, top_k=5):
        if len(self.dataset_paths) == 0:
            return []
            
        query_embedding = self.get_query_embedding(image_stream)
        similarities = cosine_similarity(query_embedding, self.dataset_embeddings)
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "image_path": self.dataset_paths[idx],
                "score": float(similarities[0][idx])
            })
        return results

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
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# Use a separate bucket for dataset images to keep them organized
fs = GridFSBucket(db, bucket_name='clothes')
dataset_fs = GridFSBucket(db, bucket_name='dataset_cache') 

search_engine = FashionSearchEngine(
    model_path="fashion_model_v1.pth", 
    index_path="fashion_index_v1.pkl",
    device="cpu"
)

def find_file_on_disk(target_path):
    """
    If the path is wrong, this searches the system for the file.
    It returns the absolute path if found, or None.
    """
    attempts = [
        target_path,
        os.path.join(os.getcwd(), '..', 'datasets', target_path),
        os.path.join(os.getcwd(), 'datasets', target_path),
        os.path.join(os.getcwd(), target_path)
    ]
    
    if 'img/img/' in target_path:
        fixed = target_path.replace('img/img/', 'img/')
        attempts.append(os.path.join(os.getcwd(), '..', 'datasets', fixed))
    
    for path in attempts:
        if os.path.exists(path):
            return path
            
    filename_only = os.path.basename(target_path)
    search_root = os.path.join(os.getcwd(), '..') 
    
    print(f"Deep searching for {filename_only}...")
    for root, dirs, files in os.walk(search_root):
        if filename_only in files:
            found_path = os.path.join(root, filename_only)
            print(f"FOUND FILE AT: {found_path}")
            return found_path
            
    return None

@app.route("/")
def index():
    return render_template("landing-page.html")

@app.route("/login", methods=["GET","POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        try:
            user = collection.find_one({'email': email})
            if user and check_password_hash(user["hash"], password): 
                session['email'] = email
                return redirect('/home')
            else:
                flash("Invalid email or password!")
                return render_template("login.html")
        except Exception as e:
            flash("An unexpected error occurred: " + str(e))
            return render_template("login.html")
    else:
        return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register_page():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirm")
        
        if password != confirm:
            flash("Passwords do not match!")
            return render_template("register.html")
            
        hash_pw = generate_password_hash(password)
        try:
            collection.insert_one({'email': email, 'hash': hash_pw})
            return redirect('/login')  
        except DuplicateKeyError:
            flash("Email has already been registered!")
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
    user_info = google.get("https://www.googleapis.com/oauth2/v1/userinfo").json()
    session["user"] = user_info
    
    user = collection.find_one({'email': session["user"]['email']})
    if not user:
        collection.insert_one({'email': session["user"]['email'], 'name': session["user"]['name']})
        
    session['email'] = session["user"]['email']
    return redirect('/home') 

@app.route("/home", methods=["GET", "POST"])
def home():
    email = session.get('email')
    if not email:
        return redirect('/login')
    
    user = collection.find_one({'email': email})
    wardrobe = user.get('wardrobe', []) if user else []
    return render_template("homepage.html", wardrobe=wardrobe)

@app.route("/upload", methods=["GET", "POST"])
def upload_clothes():
    email = session.get('email')
    if not email:
        return redirect('/login')
    
    if request.method == "POST":
        if 'cloth_image' not in request.files:
            flash('No file selected!')
            return render_template('upload.html')
        
        file = request.files['cloth_image']
        if file.filename == '':
            flash('No file selected!')
            return render_template('upload.html')
        
        if file:
            filename = secure_filename(file.filename)
            
            # 1. Save to MongoDB GridFS
            file.seek(0)
            file_id = fs.upload_from_stream(
                filename=filename,
                source=file,
                metadata={'email': email}
            )
            
            collection.update_one(
                {'email': email},
                {'$push': {
                    'wardrobe': {
                        'file_id': str(file_id), 
                        'filename': filename, 
                        'uploaded_at': datetime.utcnow()
                    }
                }}
            )
            
            try:
                file.seek(0) 
                similar_images = search_engine.search(file, top_k=5)
                flash('Clothes added! Showing similar styles.')
                
                return render_template(
                    'results.html', 
                    uploaded_file_id=str(file_id), 
                    similar_images=similar_images
                )
            except Exception as e:
                print(f"Search Error: {e}")
                flash('Clothes added, but search failed.')
                return redirect('/home')

    return render_template('upload.html')

@app.route('/clothes/<file_id>')
def serve_clothes(file_id):
    try:
        grid_out = fs.open_download_stream(ObjectId(file_id))
        return send_file(
            io.BytesIO(grid_out.read()),
            mimetype=grid_out.content_type or 'image/jpeg',
            as_attachment=False,
            download_name=grid_out.filename
        )
    except:
        return send_file('static/assets/placeholder.jpg')

@app.route('/dataset_img/<path:filename>')
def serve_dataset_image(filename):
    try:
        file_doc = db['dataset_cache.files'].find_one({"filename": filename})
        if file_doc:
            grid_out = dataset_fs.open_download_stream(file_doc['_id'])
            return send_file(
                io.BytesIO(grid_out.read()),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=filename.split('/')[-1]
            )
    except Exception as e:
        print(f"Cache lookup error: {e}")

    print(f"[Cache Miss] Looking for {filename} on disk...")
    real_path = find_file_on_disk(filename)
    
    if real_path and os.path.exists(real_path):
        try:
            with open(real_path, 'rb') as f:
                dataset_fs.upload_from_stream(
                    filename=filename, 
                    source=f
                )
            print(f"[Cache Set] Uploaded {filename} to MongoDB.")
        except Exception as e:
            print(f"Failed to cache to DB: {e}")

        # Serve the file from disk this one time
        return send_file(real_path)
    else:
        print(f"[Fatal] Could not find {filename} anywhere.")
        return "Image not found", 404


@app.route("/search", methods=["GET", "POST"])
def visual_search_only():
    """
    Allows user to upload an image and search WITHOUT saving to DB.
    """
    if request.method == "POST":
        if 'cloth_image' not in request.files:
            flash('No file selected!')
            return render_template('search.html')
        
        file = request.files['cloth_image']
        if file.filename == '':
            flash('No file selected!')
            return render_template('search.html')
        
        if file:
            try:
                image_data = file.read()
                encoded_image = base64.b64encode(image_data).decode('utf-8')
                file.seek(0) 
                similar_images = search_engine.search(file, top_k=5)
                
                return render_template(
                    'results.html', 
                    similar_images=similar_images,
                    query_image_b64=encoded_image, 
                    uploaded_file_id=None 
                )
            except Exception as e:
                print(f"Search Error: {e}")
                flash('Search failed. Please try again.')
                return render_template('search.html')

    return render_template('search.html')

@app.route("/logout")
def logout():
    session.clear()
    return redirect('/') 

if __name__ == "__main__":
    app.run(debug=True)