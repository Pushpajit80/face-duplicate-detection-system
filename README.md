# Face Duplicate Detection System

A Python-based web application designed to identify potential duplicate identities
using facial recognition and Optical Character Recognition (OCR).

The system processes uploaded images and documents, extracts relevant information,
generates facial embeddings, and compares faces to identify potential duplicate
records.

## 🚀 Features

- **Face Detection and Recognition:** Uses InsightFace to detect faces and
  generate facial embeddings.
- **Duplicate Face Matching:** Compares facial embeddings using cosine similarity.
- **OCR-Based Information Extraction:** Uses Tesseract OCR to extract text
  from supported images and documents.
- **Document Processing:** Supports PDF processing using PyMuPDF.
- **Web Interface:** Provides a Flask-based interface for uploading files
  and viewing detection results.
- **Result Visualization:** Presents duplicate detection results through
  the web interface.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Web Framework:** Flask
- **Face Recognition:** InsightFace
- **Computer Vision:** OpenCV
- **OCR:** Tesseract OCR
- **PDF Processing:** PyMuPDF
- **Numerical Computing:** NumPy
- **Data Storage:** CSV and Pickle

## 📂 Project Structure

```text
face-duplicate-detection-system/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   ├── dataset_a/
│   │   ├── images/
│   │   ├── metadata.csv
│   │   ├── voterids.csv
│   │   └── embeddings.pkl
│   │
│   ├── dataset_b/
│   └── uploads/
│
└── .gitignore
```

*Note: Update this structure to match your actual project files.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/face-duplicate-detection-system.git
```

Navigate to the project directory:

```bash
cd face-duplicate-detection-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure that Tesseract OCR is installed separately and configured
correctly on your system if your application uses it.

### 4. Run the Application

```bash
python app.py
```

Open the local URL displayed in your terminal, typically:

http://127.0.0.1:5000/

## 🔍 How It Works

1. **Upload:** The user uploads images or supported documents through the web interface.
2. **Processing:** The application processes the uploaded files and prepares
   images for analysis.
3. **Face Analysis:** InsightFace detects faces and generates facial embeddings.
4. **Similarity Matching:** The system compares facial embeddings using cosine
   similarity to identify potential matches.
5. **Information Extraction:** Tesseract OCR extracts available text from
   supported images or documents.
6. **Results:** The application displays potential duplicate matches through
   the web interface.

## 🎯 Use Cases

- Duplicate image and identity record analysis.
- Face-matching experiments.
- Computer vision and machine learning projects.
- Document-processing workflow demonstrations.

## 🔮 Future Improvements

- Add a database for persistent record management.
- Improve duplicate matching with configurable similarity thresholds.
- Add downloadable reports and detailed match scores.
- Improve handling of low-quality images and multiple faces.
- Add authentication and secure file management.
- Evaluate accuracy using a labeled test dataset.

## 🔐 Privacy and Responsible Use

Facial recognition involves sensitive biometric information.

Use this project only with authorized data and appropriate consent.
Avoid uploading real identity documents or biometric information to public
repositories. Do not commit API keys, passwords, private datasets, or
personally identifiable information.

Similarity matches should be treated as potential matches rather than
conclusive proof that two records belong to the same person.

## 👨‍💻 Author

Pushpajit Kumar Singh

GitHub: https://github.com/Pushpajit80

This project is available under the MIT License if the repository includes
an MIT LICENSE file. Otherwise, specify the license you choose before
redistributing the project.
