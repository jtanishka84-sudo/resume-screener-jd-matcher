import streamlit as st
import pdfplumber
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher

# ---------- CONFIG ----------
SKILL_CATEGORIES = {
    "Programming": ["python", "java", "c++", "c", "sql", "r", "scala", "javascript", "html", "css"],
    "ML / AI": ["machine learning", "deep learning", "artificial intelligence", "nlp",
                "natural language processing", "computer vision", "data science",
                "data analysis", "data visualization", "statistics", "mathematics"],
    "Frameworks & Libraries": ["tensorflow", "pytorch", "keras", "scikit-learn", "opencv",
                                "pandas", "numpy", "matplotlib", "seaborn", "flask",
                                "django", "fastapi", "react", "node.js"],
    "Tools & Platforms": ["aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
                           "linux", "tableau", "power bi", "excel", "mysql", "mongodb",
                           "postgresql"],
    "Core CS & Soft Skills": ["data structures", "algorithms", "rest api", "agile",
                               "communication", "teamwork", "leadership"],
}

SKILLS_LIST = [skill for skills in SKILL_CATEGORIES.values() for skill in skills]

SKILL_TO_CATEGORY = {}
for category, skills in SKILL_CATEGORIES.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill] = category


# ---------- MODEL LOADERS ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_category_model():
    model = pickle.load(open("category_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer


@st.cache_resource
def load_nlp_and_matcher():
    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in SKILLS_LIST]
    matcher.add("SKILLS", patterns)
    return nlp, matcher


embedding_model = load_embedding_model()
category_model, category_vectorizer = load_category_model()
nlp, matcher = load_nlp_and_matcher()


# ---------- CORE FUNCTIONS ----------
def extract_text_from_pdf(file):
    text = ""
    word_count = 0
    page_count = 0
    with pdfplumber.open(file) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                word_count += len(page_text.split())
    return text, page_count, word_count


def extract_skills(text, nlp, matcher):
    doc = nlp(text.lower())
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found.add(span.text)
    return found


def extract_sections(text):
    section_headers = {
        "education": ["education", "academic background"],
        "experience": ["experience", "work experience", "employment", "professional experience"],
        "projects": ["projects", "academic projects", "personal projects",
                      "ml / ai projects", "ml/ai projects"],
        "skills": ["skills", "technical skills", "core competencies"],
        "certifications": ["certifications", "certificates", "licenses"],
        "achievements": ["achievements", "awards", "honors"],
        "summary": ["summary", "objective", "profile"],
        "extracurricular": ["extracurricular", "extracurricular & leadership", "leadership"],
    }

    sections = {"other": []}
    current_section = "other"

    for line in text.split("\n"):
        line_clean = re.sub(r'[^a-z\s]', '', line.strip().lower())
        matched_header = None
        for header, variants in section_headers.items():
            if line_clean in variants or any(line_clean.startswith(v) for v in variants):
                matched_header = header
                break
        if matched_header:
            current_section = matched_header
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def predict_category(resume_text, model, vectorizer):
    cleaned = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', resume_text).lower()
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = max(proba) * 100
    return prediction, confidence


def calculate_semantic_similarity(resume_text, job_description, model):
    embeddings = model.encode([resume_text, job_description])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100


def category_breakdown(resume_skills, jd_skills):
    rows = []
    for category in SKILL_CATEGORIES:
        jd_in_cat = {s for s in jd_skills if SKILL_TO_CATEGORY.get(s) == category}
        resume_in_cat = {s for s in resume_skills if SKILL_TO_CATEGORY.get(s) == category}
        if jd_in_cat:
            coverage = len(resume_in_cat & jd_in_cat) / len(jd_in_cat) * 100
        else:
            coverage = None
        rows.append({
            "category": category,
            "jd_skills": jd_in_cat,
            "resume_skills": resume_in_cat,
            "coverage": coverage,
        })
    return rows


def calculate_overall_skill_score(breakdown):
    scored = [row["coverage"] for row in breakdown if row["coverage"] is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


# ---------- ATS CHECKER ----------
def run_ats_checks(text, sections, page_count, word_count):
    """
    Rule-based ATS (Applicant Tracking System) compatibility checks.
    Each check returns (passed: bool, pass_message, fail_message).
    """
    checks = []

    has_email = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))
    has_phone = bool(re.search(r'(\+?\d[\d\-\s]{8,}\d)', text))
    checks.append((has_email, "Email address found", "No email address detected"))
    checks.append((has_phone, "Phone number found", "No phone number detected"))

    required_sections = ["education", "skills"]
    for sec in required_sections:
        present = sec in sections
        checks.append((present, f"'{sec.title()}' section detected",
                        f"No clear '{sec.title()}' section header found"))

    length_ok = 150 <= word_count <= 1200
    checks.append((length_ok,
                    f"Resume length is reasonable ({word_count} words)",
                    f"Resume may be too short or too long ({word_count} words; aim for 300-900)"))

    pages_ok = page_count <= 2
    checks.append((pages_ok,
                    f"Page count is appropriate ({page_count} page{'s' if page_count > 1 else ''})",
                    f"Resume has {page_count} pages; ATS and recruiters generally prefer 1-2 pages"))

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    short_lines = [l for l in lines if len(l.split()) <= 20]
    bullet_ratio = len(short_lines) / len(lines) if lines else 0
    bullets_ok = bullet_ratio > 0.5
    checks.append((bullets_ok,
                    "Content appears to use concise, scannable lines",
                    "Content appears to be in dense paragraphs; ATS and recruiters prefer bullet points"))

    has_skills_content = "skills" in sections and len(sections["skills"]) > 10
    checks.append((has_skills_content,
                    "Skills section has identifiable content",
                    "Skills section is missing or very sparse"))

    passed_count = sum(1 for ok, _, _ in checks if ok)
    score = passed_count / len(checks) * 100
    return score, checks


def generate_report(skill_score, semantic_score, final_score, matched, missing,
                     predicted_category, confidence, ats_score):
    skill_score_display = f"{skill_score:.1f}%" if skill_score is not None else "N/A (JD listed no recognized skills)"
    report = f"""RESUME SCREENING REPORT
========================

Predicted Resume Category: {predicted_category} (Confidence: {confidence:.1f}%)

Overall Match Score: {final_score:.1f}%
Skill Coverage Score: {skill_score_display}
Semantic Similarity Score: {semantic_score:.1f}%
ATS Compatibility Score: {ats_score:.1f}%

Matched Skills ({len(matched)}):
{', '.join(sorted(matched)) if matched else 'None'}

Missing Skills ({len(missing)}):
{', '.join(sorted(missing)) if missing else 'None'}

Recommendation:
{"Strong match! Consider applying." if final_score > 60 else "Consider strengthening the missing skills above, or highlighting relevant experience that demonstrates them."}
"""
    return report


def make_radar_chart(breakdown):
    categories = [row["category"] for row in breakdown]
    jd_values = []
    resume_values = []
    for row in breakdown:
        total_in_cat = len(SKILL_CATEGORIES[row["category"]])
        jd_values.append(len(row["jd_skills"]) / total_in_cat * 100)
        resume_values.append(len(row["resume_skills"]) / total_in_cat * 100)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    jd_values += jd_values[:1]
    resume_values += resume_values[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, resume_values, linewidth=2, label="Your resume", color="#378ADD")
    ax.fill(angles, resume_values, alpha=0.2, color="#378ADD")
    ax.plot(angles, jd_values, linewidth=2, label="Job description", color="#D85A30")
    ax.fill(angles, jd_values, alpha=0.15, color="#D85A30")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


# ---------- UI HELPERS ----------
def metric_card(label, value, highlight=False):
    bg = "#E6F1FB" if highlight else "#F0F2F6"
    color = "#0C447C" if highlight else "#1a1a1a"
    label_color = "#185FA5" if highlight else "#6b6b6b"
    st.markdown(
        f"""
        <div style="background:{bg}; border-radius:8px; padding:1rem; height:100%;">
            <div style="font-size:13px; color:{label_color};">{label}</div>
            <div style="font-size:24px; font-weight:600; margin-top:4px; color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_bar(label, pct, note=""):
    pct_display = f"{pct:.0f}%" if pct is not None else "N/A"
    width = pct if pct is not None else 0
    st.markdown(
        f"""
        <div style="margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                <span>{label}</span>
                <span style="color:#6b6b6b;">{pct_display}{f' &middot; {note}' if note else ''}</span>
            </div>
            <div style="height:6px; border-radius:4px; background:#e6e6e6; overflow:hidden;">
                <div style="height:100%; width:{width}%; background:#378ADD;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def skill_pills(skills, kind="matched"):
    if not skills:
        st.write("None")
        return
    if kind == "matched":
        bg, fg = "#EAF3DE", "#27500A"
    else:
        bg, fg = "#FCEBEB", "#791F1F"
    pills_html = "".join(
        f'<span style="background:{bg}; color:{fg}; font-size:12px; padding:4px 10px; '
        f'border-radius:8px; margin:2px; display:inline-block;">{s}</span>'
        for s in sorted(skills)
    )
    st.markdown(f'<div>{pills_html}</div>', unsafe_allow_html=True)


def ats_score_ring(score):
    color = "#1D9E75" if score >= 75 else ("#EF9F27" if score >= 50 else "#E24B4A")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:center; flex-direction:column;">
            <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="50" fill="none" stroke="#e6e6e6" stroke-width="12"/>
                <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="12"
                        stroke-dasharray="{score * 3.14}, 314" stroke-linecap="round"
                        transform="rotate(-90 60 60)"/>
                <text x="60" y="65" text-anchor="middle" font-size="22" font-weight="600" fill="{color}">{score:.0f}%</text>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Resume Screener", layout="wide")

st.title("Resume screener & JD matcher")
st.write(
    "Upload your resume (PDF) and paste a job description. "
    "Get an ML-predicted job category, skill coverage by category, "
    "semantic similarity to the JD, and an ATS compatibility check."
)

col_upload, col_jd = st.columns(2)
with col_upload:
    resume_file = st.file_uploader("Upload resume (PDF)", type="pdf")
with col_jd:
    job_description = st.text_area("Paste job description here", height=180)

analyze = st.button("Analyze", type="primary")

if analyze:
    if resume_file is None or job_description.strip() == "":
        st.warning("Please upload a resume AND paste a job description.")
    else:
        with st.spinner("Analyzing resume..."):
            resume_text, page_count, word_count = extract_text_from_pdf(resume_file)

            predicted_category, confidence = predict_category(
                resume_text, category_model, category_vectorizer
            )

            resume_skills = extract_skills(resume_text, nlp, matcher)
            jd_skills = extract_skills(job_description, nlp, matcher)

            breakdown = category_breakdown(resume_skills, jd_skills)
            skill_score = calculate_overall_skill_score(breakdown)

            matched = resume_skills & jd_skills
            missing = jd_skills - resume_skills

            semantic_score = calculate_semantic_similarity(resume_text, job_description, embedding_model)

            if skill_score is not None:
                final_score = 0.5 * skill_score + 0.5 * semantic_score
            else:
                final_score = semantic_score

            sections = extract_sections(resume_text)
            ats_score, ats_checks = run_ats_checks(resume_text, sections, page_count, word_count)

        st.divider()

        # ---- Top metric cards ----
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Overall score", f"{final_score:.0f}%")
        with c2:
            metric_card("Skill coverage", f"{skill_score:.0f}%" if skill_score is not None else "N/A")
        with c3:
            metric_card("Semantic similarity", f"{semantic_score:.0f}%")
        with c4:
            metric_card("Predicted category", predicted_category, highlight=True)
            st.caption(f"Confidence: {confidence:.0f}%")

        if skill_score is None:
            st.info(
                "The job description didn't mention any skills from our tracked list, "
                "so the overall score is based only on semantic similarity."
            )

        st.markdown("")

        # ---- Skill profile + radar ----
        left, right = st.columns([1.2, 1])
        with left:
            with st.container(border=True):
                st.subheader("Skill profile by category")
                st.caption("Fraction of each category's skills found in your resume vs. requested by the JD.")
                for row in breakdown:
                    progress_bar(row["category"], row["coverage"],
                                  note="of JD requirements" if row["coverage"] is not None else "not required by JD")

        with right:
            with st.container(border=True):
                st.subheader("Resume vs JD radar")
                fig = make_radar_chart(breakdown)
                st.pyplot(fig)

        st.markdown("")

        # ---- Matched / missing pills ----
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.subheader("Matched skills")
                skill_pills(matched, kind="matched")
        with m2:
            with st.container(border=True):
                st.subheader("Missing skills")
                skill_pills(missing, kind="missing")

        st.markdown("")

        # ---- ATS Checker ----
        with st.container(border=True):
            st.subheader("ATS compatibility check")
            st.caption("Rule-based checks for formatting and structure issues that can trip up Applicant Tracking Systems.")
            ats_col1, ats_col2 = st.columns([1, 2])
            with ats_col1:
                ats_score_ring(ats_score)
            with ats_col2:
                for ok, pass_msg, fail_msg in ats_checks:
                    icon = "✅" if ok else "⚠️"
                    st.write(f"{icon} {pass_msg if ok else fail_msg}")

        st.markdown("")

        # ---- Full skill profile ----
        with st.container(border=True):
            st.subheader("Full skill profile detected in resume")
            skill_pills(resume_skills, kind="matched")

        st.markdown("")

        # ---- Resume structure ----
        with st.container(border=True):
            st.subheader("Resume structure")
            for sec, content in sections.items():
                if content:
                    with st.expander(sec.title()):
                        st.text(content)
            with st.expander("View full extracted resume text"):
                st.text(resume_text)

        st.markdown("")

        # ---- Download report ----
        report_text = generate_report(
            skill_score, semantic_score, final_score, matched, missing,
            predicted_category, confidence, ats_score
        )
        st.download_button("Download report (.txt)", report_text, file_name="resume_report.txt")