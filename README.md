# AI Resume Analyzer 🚀

An AI-powered Resume Analyzer built using Streamlit, NLP embeddings, and Dockerized cloud deployment workflows. The platform evaluates resumes against job descriptions using semantic similarity techniques and provides ATS scoring, missing skill detection, section-wise insights, and downloadable reports.

---

# 🌐 Live Demo

https://ai-resume-analyzer-j8v1.onrender.com/

---

# 📌 Features

✅ AI-powered ATS score calculation
✅ Semantic resume-job description matching
✅ Missing skills detection
✅ Section-wise resume analysis
✅ Downloadable analysis reports
✅ User authentication system
✅ Secure password hashing
✅ Resume history storage
✅ Responsive dashboard UI
✅ Dockerized deployment
✅ CI/CD automation with GitHub Actions
✅ Cloud deployment on Render

---

# 🧠 AI & NLP Capabilities

The application uses Sentence Transformers and semantic embeddings to compare resumes with job descriptions intelligently instead of relying only on keyword matching.

### Implemented AI Features:

* Semantic similarity scoring
* ATS compatibility analysis
* Skill gap detection
* Resume relevance scoring
* NLP-based matching system

---

# 🛠️ Tech Stack

## Frontend

* Streamlit
* HTML/CSS

## Backend

* Python
* SQLite

## AI / NLP

* SentenceTransformers
* Hugging Face Transformers
* Scikit-learn

## DevOps & Deployment

* Docker
* Docker Compose
* GitHub Actions
* Docker Hub
* Render

## Version Control

* Git
* GitHub

---

# 🔐 Authentication System

Implemented secure authentication with:

* User registration
* Login functionality
* Password hashing
* Persistent SQLite database storage
* Session management

---

# 🐳 Docker & Deployment

The project is fully containerized using Docker and supports automated cloud deployment workflows.

### Deployment Workflow

```bash
VS Code
↓
Git Push
↓
GitHub Actions
↓
Docker Build
↓
Docker Hub Push
↓
Render Deployment
↓
Live Application
```

---

# ⚙️ CI/CD Pipeline

Implemented GitHub Actions workflow for:

* Automated Docker image builds
* Docker Hub integration
* Continuous deployment workflow
* Automated production deployment support

---

# 📷 Application Screenshots

## 🔹 Login & Authentication

![](<screenshot/Screenshot 2026-05-02 175806.png>)


## 🔹 ATS Dashboard
![alt text](<screenshot/Screenshot 2026-05-02 175835 - Copy.png>)
![alt text](<screenshot/Screenshot 2026-05-02 180526 - Copy (2).png>)

## 🔹 Missing Skills Detection

![alt text](<screenshot/Screenshot 2026-05-02 181148 - Copy.png>)
![alt text](<screenshot/Screenshot 2026-05-02 181204.png>)
![alt text](<screenshot/Screenshot 2026-05-02 181252.png>)

## 🔹 Resume Analysis Report
![alt text](<screenshot/Screenshot 2026-05-02 182321.png>)
![alt text](<screenshot/Screenshot 2026-05-02 182327.png>)
---

# 🚀 Installation & Local Setup

## Clone Repository

```bash
git clone https://github.com/dsoni26/AI-Resume-Analyzer.git
```

## Navigate to Project

```bash
cd AI-Resume-Analyzer
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run main.py
```

---

# 🐳 Docker Setup

## Build Docker Image

```bash
docker build -t resume-analyzer .
```

## Run Docker Container

```bash
docker run -p 8501:8501 resume-analyzer
```

---

# 📈 Future Enhancements

* PostgreSQL integration
* AWS deployment
* FastAPI backend
* React frontend
* Multi-role recruiter dashboard
* Resume recommendation engine
* Interview question generation
* Analytics dashboard

---

# 💡 Key Learnings

Through this project, I gained hands-on experience in:

* AI/NLP integration
* Docker containerization
* CI/CD automation
* Cloud deployment
* Authentication systems
* Database integration
* Production debugging
* Full deployment lifecycle management

---

# 👩‍💻 Author

Dimple Soni

GitHub: https://github.com/dsoni26
