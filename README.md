<div align="center">

# 🧵 FabrAIc

**AI-powered fashion search and personal wardrobe management**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)

*Upload a photo. Find your style. Build your wardrobe.*

</div>

---

## What is FabrAIc?

FabrAIc is a full-stack fashion app that lets users visually search a fashion catalog using their own photos. Upload a clothing item, and the system finds the top 5 most similar styles from a pre-indexed dataset — powered by a fine-tuned ResNet50 model. Authenticated users can also build a persistent digital wardrobe stored in MongoDB.

---

## Features

**Visual Search** — Upload any clothing image to find the 5 most similar items from a fashion catalog. Uses 2048-dimensional ResNet50 embeddings with cosine similarity for fast, accurate matching.

**Wardrobe Management** — Save uploaded clothes to your personal wardrobe, stored in MongoDB GridFS and viewable from your homepage.

**Multi-task AI Model** — A custom ResNet50 backbone simultaneously predicts 50 clothing categories and 25 style attributes.

**Authentication** — Email/password login and registration, plus Google OAuth for one-click sign-in.

**Responsive UI** — Clean, mobile-friendly interface with custom CSS across all pages (landing, login, search, results, upload).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask, Python |
| ML | PyTorch, ResNet50, scikit-learn |
| Database | MongoDB, GridFS |
| Auth | Authlib (Google OAuth), Werkzeug |
| Frontend | Jinja2, HTML/CSS |

---

## Project Structure

```
fabraic/
├── backend.py                  # Flask app: routes, search engine, auth
├── fashion_model_v1.pth        # Trained ResNet50 weights
├── fashion_index_v1.pkl        # Pre-computed embedding index
├── scripts/
│   ├── main.ipynb              # Training pipeline & index generation (PyTorch)
│   └── app.ipynb               # Earlier experiments (TensorFlow)
├── static/
│   ├── assets/                 # Images and icons
│   ├── landing-page.css
│   ├── homepage.css
│   ├── login.css
│   ├── register.css
│   └── circular.css
└── templates/
    ├── landing-page.html
    ├── login.html
    ├── register.html
    ├── homepage.html
    ├── upload.html
    ├── search.html
    └── results.html
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install flask torch torchvision pymongo authlib python-dotenv pillow scikit-learn tqdm
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MONGO_CLIENT=mongodb://localhost:27017/userinfo
SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

### 3. Add model files

Place both files in the project root:

| File | Description |
|---|---|
| `fashion_model_v1.pth` | Trained ResNet50 weights |
| `fashion_index_v1.pkl` | Pre-computed embeddings for the catalog |

> These can be generated from scratch using `scripts/main.ipynb`.

### 4. Prepare the dataset

Place the DeepFashion dataset files in a `../datasets/` directory relative to the project root. Required files include `list_category_cloth.txt`, `list_attr_cloth.txt`, `list_eval_partition.txt`, and the `img/` folder.

### 5. Run

```bash
python backend.py
```

Visit [http://localhost:5000](http://localhost:5000).

---

## How the Search Works

```
User uploads image
        ↓
ResNet50 backbone extracts a 2048-dim feature vector
        ↓
Cosine similarity computed against pre-indexed catalog embeddings
        ↓
Top-5 matches returned and served from GridFS or disk
```

The model was fine-tuned on the DeepFashion dataset using a multi-task objective: cross-entropy loss for category classification and binary cross-entropy for attribute prediction, trained jointly.

---

## Generating the Model & Index

Open `scripts/main.ipynb` and run the cells in order:

1. **Data loading** — reads DeepFashion partition, category, and attribute files
2. **Training** — fine-tunes `MultiTaskResNet` (ResNet50 backbone + category head + attribute head) for 3 epochs
3. **Indexing** — runs inference over the full catalog to produce `fashion_index_v1.pkl`
4. **Validation** — runs sample visual searches to verify results

GPU training is supported automatically if CUDA is available.

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET/POST` | `/login` | Email/password login |
| `GET/POST` | `/register` | New account registration |
| `GET` | `/auth/google` | Google OAuth redirect |
| `GET` | `/auth/google/callback` | Google OAuth callback |
| `GET/POST` | `/home` | User homepage + wardrobe |
| `GET/POST` | `/upload` | Upload to wardrobe + search |
| `GET/POST` | `/search` | Visual search (no save) |
| `GET` | `/clothes/<file_id>` | Serve user-uploaded image |
| `GET` | `/dataset_img/<path>` | Serve catalog image (with caching) |
| `GET` | `/logout` | Clear session |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## Roadmap

- [ ] Live demo deployment
- [ ] Outfit compatibility scoring (upper + lower body pairing)
- [ ] Attribute-based filtering on search results
- [ ] Shareable wardrobe profiles

---

<div align="center">
<sub>Built with PyTorch, Flask, and a love for fashion.</sub>
</div>