# PluraAI

### AI-Powered Pneumonia Detection & Intelligent Chest X-ray Analysis

PluraAI is an AI-powered healthcare application designed to assist in the analysis of chest X-ray images for pneumonia detection.

The system combines a **Vision Transformer (ViT)-based deep learning model** with a **Django backend** and an interactive frontend to provide an end-to-end workflow for uploading and analyzing chest X-rays.

Along with pneumonia prediction, PluraAI provides supporting features such as **AI-powered assistance, prediction results, image processing, and an interactive web interface** for exploring the analysis.

---

# Features

- **Chest X-ray Analysis**
  - Upload a chest X-ray image for analysis.

- **ViT-based Pneumonia Detection**
  - Uses a trained Vision Transformer model for image classification.
  - The trained model is stored using Git LFS.

- **Image Preprocessing**
  - Preprocesses uploaded X-ray images before inference.

- **Django Backend**
  - Handles API requests, inference-related operations, image uploads, and backend services.

- **AI-powered Assistant**
  - Provides an interactive AI assistant using the OpenRouter API.

- **Interactive Web Interface**
  - Provides a user-friendly interface for interacting with the system.

- **Prediction Results**
  - Displays the model's prediction after processing the uploaded image.


---

# Project Structure

```text
PluraAI/
│
├── frontend/
│   └── pnumoni_Ui/
│       ├── images/
│       ├── static/
│       ├── templates/
│       ├── app.py
│       ├── requirements.txt
│       └── .env.example
│
├── PluraAIBackend/
│   └── backend/
│       ├── backend/
│       ├── inference/
│       ├── utils/
│       ├── uploads/
│       ├── manage.py
│       ├── requirements.txt
│       └── .env.example
│
├── dataset.py
├── model.py
├── preprocess.py
├── vitcal_best (1).pth
├── .gitignore
└── .gitattributes
```


---

# Technologies Used

### Frontend

- Python
- Flask
- HTML
- CSS
- JavaScript

### Backend

- Django
- Django REST Framework
- Python
- MongoDB

### Machine Learning

- PyTorch
- Torchvision
- Vision Transformer (ViT)
- OpenCV
- NumPy
- Pillow

### AI Assistant

- OpenRouter API

### Version Control

- Git
- Git LFS


---

# Getting Started

Follow the steps below to run PluraAI locally.


## 1. Prerequisites

Make sure the following are installed:

- Python 3.10 or 3.11
- Git
- Git LFS

Check Python:

```bash
python --version
```

Check Git:

```bash
git --version
```

Check Git LFS:

```bash
git lfs version
```

If Git LFS is installed but not initialized:

```bash
git lfs install
```


---

# 2. Clone the Repository

Clone the repository:

```bash
git clone https://github.com/laxmisahithi10/PluraAI.git
```

Move into the project directory:

```bash
cd PluraAI
```

Download the large model file managed by Git LFS:

```bash
git lfs pull
```

This downloads the trained model:

```text
vitcal_best (1).pth
```

The trained model is stored using Git LFS because of its large file size.


---

# 3. Configure Environment Variables

The project uses environment variables for sensitive information such as:

- API keys
- Database credentials
- Django configuration

> **Do not commit `.env` files to GitHub.**

Use the provided `.env.example` files as templates.


## Frontend Environment

Navigate to the frontend:

```bash
cd frontend/pnumoni_Ui
```

Create the `.env` file.

### Windows

```powershell
copy .env.example .env
```

Open `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Then return to the project root:

```bash
cd ../..
```


## Backend Environment

Navigate to the backend:

```bash
cd PluraAIBackend/backend
```

Create the `.env` file.

### Windows

```powershell
copy .env.example .env
```

Configure the required backend variables:

```env
DJANGO_SECRET_KEY=your_django_secret_key
MONGO_URI=your_mongodb_connection_string
MONGO_DB_NAME=plura_db
```

Then return to the project root:

```bash
cd ../..
```


---

# ⚙️ 4. Run the Django Backend

Open a terminal and navigate to:

```bash
cd PluraAIBackend/backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run database migrations:

```bash
python manage.py migrate
```

Create the uploads directory if it does not already exist:

```powershell
mkdir uploads
```

Start the Django development server:

```bash
python manage.py runserver
```

The backend will run at:

```text
http://127.0.0.1:8000
```


---

# 🌐 5. Run the Frontend

Open a **new terminal**.

Navigate to:

```bash
cd frontend/pnumoni_Ui
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Make sure the `.env` file contains your OpenRouter API key.

Start the frontend:

```bash
python app.py
```

The frontend will run at the address displayed in the terminal, typically:

```text
http://127.0.0.1:5000
```


---

# Application Workflow

```text
              ┌─────────────────────┐
              │  User uploads       │
              │  Chest X-ray        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Image Preprocessing │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Vision Transformer  │
              │      Model          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Pneumonia           │
              │ Prediction          │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Django Backend /    │
              │ API                 │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Web Interface &     │
              │ AI Assistant        │
              └─────────────────────┘
```


---

# Model

PluraAI uses a trained **Vision Transformer (ViT)** model for pneumonia detection.

The trained model is:

```text
vitcal_best (1).pth
```

The model is approximately **380 MB** in size and is therefore stored using **Git Large File Storage (Git LFS)** instead of standard Git.

After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

This downloads the actual model file required for inference.


---

# Dependencies

The project maintains separate dependency files for the frontend and backend.

## Backend

Dependency file:

```text
PluraAIBackend/backend/requirements.txt
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Frontend

Dependency file:

```text
frontend/pnumoni_Ui/requirements.txt
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```
