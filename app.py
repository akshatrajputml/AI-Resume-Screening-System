import streamlit as st
import pdfplumber
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("AI Resume Screening & ATS System")
st.write("Upload your resume and get AI-powered analysis.")

# =====================================================
# SKILLS DATABASE
# =====================================================

skills_list = [

    "python",
    "machine learning",
    "deep learning",
    "sql",
    "tensorflow",
    "pytorch",
    "nlp",
    "data analysis",
    "data science",
    "computer vision",
    "flask",
    "streamlit",
    "power bi",
    "excel",
    "scikit-learn",
    "pandas",
    "numpy",
    "statistics",
    "matplotlib",
    "seaborn"

]

# =====================================================
# JOB DATABASE
# =====================================================

jobs = {

    "Data Scientist":
    """
    python machine learning statistics pandas numpy
    data visualization sql data analysis
    """,

    "AI Engineer":
    """
    deep learning tensorflow pytorch
    computer vision nlp neural networks
    """,

    "Data Analyst":
    """
    sql excel power bi data analysis
    dashboard visualization statistics
    """,

    "ML Engineer":
    """
    machine learning deployment flask
    streamlit python scikit-learn mlops
    """
}

# =====================================================
# JOB SKILL REQUIREMENTS
# =====================================================

job_skill_requirements = {

    "Data Scientist": [
        "python",
        "machine learning",
        "sql",
        "pandas",
        "numpy",
        "data analysis"
    ],

    "AI Engineer": [
        "python",
        "deep learning",
        "tensorflow",
        "pytorch",
        "computer vision",
        "nlp"
    ],

    "Data Analyst": [
        "sql",
        "excel",
        "power bi",
        "data analysis",
        "statistics"
    ],

    "ML Engineer": [
        "python",
        "machine learning",
        "flask",
        "streamlit",
        "scikit-learn"
    ]
}

# =====================================================
# DOCX READER
# =====================================================

def extract_text_from_docx(docx_file):

    doc = Document(docx_file)

    full_text = []

    for para in doc.paragraphs:

        full_text.append(para.text)

    return '\n'.join(full_text)

# =====================================================
# PDF READER
# =====================================================

def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text

# =====================================================
# SKILL EXTRACTION
# =====================================================

def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_list:

        if skill.lower() in text:

            found_skills.append(skill)

    return found_skills

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)

# =====================================================
# MAIN LOGIC
# =====================================================

if uploaded_file is not None:

    file_name = uploaded_file.name

    if file_name.endswith(".pdf"):

        resume_text = extract_text_from_pdf(uploaded_file)

    elif file_name.endswith(".docx"):

        resume_text = extract_text_from_docx(uploaded_file)

    skills_found = extract_skills(resume_text)

    job_titles = list(jobs.keys())

    job_descriptions = list(jobs.values())

    documents = [resume_text] + job_descriptions

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )

    scores = similarity[0]

    recommended_jobs = sorted(

        zip(job_titles, scores),

        key=lambda x: x[1],

        reverse=True

    )

    best_job = recommended_jobs[0][0]

    required_skills = job_skill_requirements[best_job]

    matched_skills = list(
        set(skills_found).intersection(
            set(required_skills)
        )
    )

    missing_skills = list(
        set(required_skills) - set(skills_found)
    )

    ats_score = (
        len(matched_skills) /
        len(required_skills)
    ) * 100

    st.success("Resume Processed Successfully")

    st.subheader("Recommended Job Role")
    st.write(best_job)

    st.subheader("ATS Score")

    st.progress(int(ats_score))

    st.write(f"{round(ats_score,2)}%")

    st.subheader("Skills Found")
    st.write(skills_found)

    st.subheader("Matched Skills")
    st.write(matched_skills)

    st.subheader("Missing Skills")
    st.write(missing_skills)

    st.subheader("Improvement Suggestions")

    if len(missing_skills) > 0:

        st.write(
            "You should improve or add these skills:"
        )

        for skill in missing_skills:

            st.write("-", skill)

    else:

        st.success(
            "Excellent Resume for this role!"
        )

    st.subheader("Job Match Scores")

    for job, score in recommended_jobs:

        st.write(
            f"{job} --> {round(score*100,2)}%"
        )
