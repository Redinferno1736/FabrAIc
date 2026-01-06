# FabrAIc - AI Outfit Recommendation System ✨

FabrAIc is a full-stack web application that uses deep learning for **fashion image search**, **outfit recommendations**, and **personal wardrobe management**. Users can upload clothes, search for similar styles from a catalog using ResNet50 features, and build a digital wardrobe with MongoDB storage.


## 🚀 Key Features

- **Visual Search** - Upload an image to find **top-5 similar outfits** from a pre-indexed fashion catalog using cosine similarity on ResNet50 embeddings
- **Wardrobe Management** - Authenticated users upload and store clothes in MongoDB GridFS, view personal wardrobe on homepage
- **AI Models** - Multi-task ResNet50 predicts **50 categories** and **25 attributes**; pre-trained models power search
- **Authentication** - Email/password login/register + **Google OAuth** integration
- **Responsive UI** - Modern templates with CSS for desktop/mobile (landing page, results, upload pages)

## 🛠 Tech Stack

| Component | Technologies |
|-----------|--------------|
| **Backend** | Flask, PyTorch, MongoDB (GridFS), scikit-learn |
| **Frontend** | HTML/CSS (Tailwind-inspired), Jinja2 templates |
| **ML** | ResNet50 (pretrained), cosine similarity search |
| **Deployment** | Local Flask server; Vercel/Render compatible |

## 📁 Directory Structure

```

redinferno1736-fabraic/
├── backend.py              \# Main Flask app with search engine
├── scripts/
│   ├── app.ipynb          \# Data processing, model training notebooks
│   └── main.ipynb         \# Feature extraction, indexing
├── static/                 \# CSS stylesheets
└── templates/              \# HTML pages (login, search, results, etc.)

```

## ⚡ Quick Setup & Run

1. **Install dependencies**:
   ```bash
   pip install flask torch torchvision pymongo authlib python-dotenv pillow scikit-learn
```

2. **Set environment variables** (create `.env`):

```
MONGO_URI=mongodb://localhost:27017/fabraic
SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

3. **Place models** in root:
    - `fashionmodelv1.pth` (ResNet50 model)
    - `fashionindexv1.pkl` (search index)
    - Generate via `scripts/main.ipynb`
4. **Datasets**: Ensure `../datasets/` has images/index files from notebooks
5. **Run**:

```bash
python backend.py
```

Visit `http://localhost:5000`

## 🧠 Models \& Data

- **Trained** on fashion dataset (~thousands images) with categories/attributes from `listcategorycloth.txt`, `listattrcloth.txt`
- **Index**: 4096-dim embeddings for catalog images
- **Search flow**: Query → ResNet50 → Cosine similarity → Top-K results served from GridFS/disk cache


## 🎯 Live Demo

Coming soon! 🚀

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---


