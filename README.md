# Resume Screener & JD Matcher

An NLP and ML-based tool that analyzes how well a resume matches a job description, 
predicts the resume's job category, checks ATS (Applicant Tracking System) compatibility, 
and provides a detailed skill-gap breakdown by category.

## Live Demo


## Features
- **PDF resume parsing** using pdfplumber
- **Resume category prediction** — a Logistic Regression classifier (TF-IDF features) 
  trained on ~960 labeled resumes across 25 job categories, achieving 99.5% test accuracy
- **Skill extraction** using spaCy's PhraseMatcher, organized into 5 categories 
  (Programming, ML/AI, Frameworks & Libraries, Tools & Platforms, Core CS & Soft Skills)
- **Category-wise skill coverage** — shows what fraction of a JD's required skills 
  (per category) appear in the resume, visualized with progress bars and a radar chart
- **Semantic similarity** using sentence-transformers (all-MiniLM-L6-v2) and cosine 
  similarity, to capture overall contextual match beyond exact keywords
- **ATS compatibility checker** — rule-based checks for contact info, section headers, 
  resume length, page count, and bullet-point formatting
- **Resume section detection** (Education, Experience, Projects, Skills, etc.)
- **Downloadable screening report**

## Tech Stack
- Python, Streamlit
- pdfplumber (PDF text extraction)
- spaCy (PhraseMatcher for skill extraction)
- sentence-transformers + scikit-learn (semantic similarity via cosine similarity)
- scikit-learn + TF-IDF (resume category classification)
- matplotlib (radar chart visualization)

## How It Works
1. User uploads a resume (PDF) and pastes a job description
2. Text is extracted from the PDF
3. A trained classifier predicts the resume's job category
4. Skills are extracted from both resume and JD using spaCy PhraseMatcher, grouped 
   into 5 categories
5. Skill coverage is computed per category and averaged
6. Sentence embeddings compute overall semantic similarity between resume and JD
7. Overall score = 50% skill coverage + 50% semantic similarity
8. Rule-based checks evaluate ATS compatibility
9. Results are displayed with visualizations, and a report can be downloaded

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Model Training
The category classifier was trained separately using `train_model.py` on the 
`Resume.csv` dataset (Kaggle). To retrain: python train_model.py

## Limitations & Future Improvements
- Skill list is curated and may not cover all domains/technologies
- Category classifier trained on a relatively small (~960 resume) dataset; 99.5% 
  accuracy may indicate some overfitting to this specific dataset's style
- Could add multi-resume ranking for recruiter-style comparison
- Could add LLM-based personalized improvement suggestions
- Currently supports PDF only (no DOCX)
