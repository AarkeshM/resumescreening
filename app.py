import os
import warnings
import re
import spacy
import numpy as np
import pdfplumber
import docx
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# ✅ Connect to MongoDB
MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)
db = client["ResumeDB"]
collection = db["Resumes"]

# ✅ Load NLP Model
nlp = spacy.load("en_core_web_sm")

# ✅ Load Sentence Transformer for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Function to extract text from PDF
def extract_text_from_pdf(file):
    """Extracts text from a PDF file and suppresses warnings."""
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")

        with pdfplumber.open(file) as pdf:
            text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
        
        return text if text else "No text found in PDF"
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# ✅ Function to extract text from DOCX
def extract_text_from_docx(file):
    """Extracts text from a DOCX file."""
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

# ✅ Function to preprocess text
def preprocess_text(text):
    """Removes special characters and stopwords."""
    text = re.sub(r'\W+', ' ', text.lower())
    return " ".join([token.lemma_ for token in nlp(text) if not token.is_stop])

# ✅ Function to generate embeddings
def get_embeddings(text):
    return model.encode(text)

# ✅ API to upload resumes & store in MongoDB
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        file = request.files.get("resume")

        if not file:
            return jsonify({"error": "No file provided"}), 400

        filename = secure_filename(file.filename)
        file_extension = filename.split(".")[-1].lower()

        # Extract text based on file type
        if file_extension == "pdf":
            resume_text = extract_text_from_pdf(file)
        elif file_extension == "docx":
            resume_text = extract_text_from_docx(file)
        elif file_extension == "txt":
            resume_text = file.read().decode("utf-8")
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        cleaned_text = preprocess_text(resume_text)
        embedding = get_embeddings(cleaned_text).tolist()  # Convert to list for MongoDB storage

        # ✅ Store in MongoDB
        resume_data = {
            "filename": filename,
            "content": resume_text,
            "embedding": embedding
        }
        collection.insert_one(resume_data)

        return jsonify({"message": "Resume uploaded and stored successfully", "filename": filename})
    
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

# ✅ API to calculate similarity scores
@app.route('/rank_resumes', methods=['POST'])
def rank_resumes():
    try:
        job_description = request.form.get("job_description", "")

        if not job_description:
            return jsonify({"error": "Job description is required"}), 400

        cleaned_job_desc = preprocess_text(job_description)
        job_embedding = get_embeddings(cleaned_job_desc)

        # ✅ Retrieve stored resumes from MongoDB
        resumes = list(collection.find({}, {"filename": 1, "embedding": 1}))

        if not resumes:
            return jsonify({"error": "No resumes found"}), 400

        similarity_scores = []
        
        for resume in resumes:
            resume_embedding = np.array(resume["embedding"])
            score = np.dot(job_embedding, resume_embedding) / (np.linalg.norm(job_embedding) * np.linalg.norm(resume_embedding))
            similarity_scores.append({"resume": resume["filename"], "score": round(score * 100, 2)})

        similarity_scores.sort(key=lambda x: x["score"], reverse=True)

        return jsonify({"ranked_resumes": similarity_scores})

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
