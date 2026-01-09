# FabrAIc - AI Outfit Recommendation System ✨

FabrAIc is a full-stack web application that uses deep learning for **fashion image search**, **outfit recommendations**, and **personal wardrobe management**. Users can upload clothes, search for similar styles from a catalog using ResNet50 features, and build a digital wardrobe with MongoDB storage.

## 🚀 Key Features

* **Visual Search** - Upload an image to find **top-5 similar outfits** from a pre-indexed fashion catalog using cosine similarity on ResNet50 embeddings.
* **Wardrobe Management** - Authenticated users upload and store clothes in MongoDB GridFS; personal wardrobes are viewable on the homepage.
* **AI Models** - A multi-task ResNet50 architecture predicts **50 categories** and **25 attributes**.
* **Authentication** - Email/password login and registration plus **Google OAuth** integration.
* **Responsive UI** - Modern templates with CSS for desktop and mobile, including landing, results, and upload pages.

## 🛠 Tech Stack

| Component | Technologies |
|:----------|:-------------|
| **Backend** | Flask, PyTorch, MongoDB (GridFS), scikit-learn |
| **Frontend** | HTML/CSS (Tailwind-inspired), Jinja2 templates |
| **ML** | ResNet50 (pretrained), Cosine Similarity search |

## 📁 Directory Structure
```
redinferno1736-fabraic/
├── backend.py                  # Main Flask app with search engine logic
├── fashion_model_v1.pth        # Trained ResNet50 model weights
├── fashion_index_v1.pkl        # Pre-computed search index
├── scripts/
│   └── main.ipynb              # Feature extraction, training, and indexing
├── static/                     # CSS stylesheets and local assets
└── templates/                  # HTML pages (login, search, results, etc.)
```

## ⚡ Quick Setup & Run

### 1. Install Dependencies
```bash
pip install flask torch torchvision pymongo authlib python-dotenv pillow scikit-learn tqdm
```

### 2. Set Environment Variables

Create a `.env` file in the root directory:
```text
MONGO_URI=mongodb://localhost:27017/userinfo
SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

### 3. Place Models in Root

* `fashion_model_v1.pth`: The trained ResNet50 model.
* `fashion_index_v1.pkl`: The generated search index.
* Note: These can be generated via `scripts/main.ipynb`.

### 4. Datasets

Ensure your `../datasets/` directory contains the required images and index files referenced in the notebooks.

### 5. Run the Application
```bash
python backend.py
```

Visit `http://localhost:5000`.

## 🧠 Models & Data

* **Training**: The model is trained on a fashion dataset utilizing categories and attributes from `list_category_cloth.txt` and `list_attr_cloth.txt`.
* **Index**: Uses 2048-dimensional embeddings (ResNet50 backbone) for catalog images.
* **Search Flow**: Query → ResNet50 Feature Extraction → Cosine Similarity → Top-K results served from GridFS or disk.

## 🎯 Live Demo

Coming soon! 🚀

## 🤝 Contributing

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
