import base64
import hashlib
import hmac
import os
import re
import sqlite3
import struct

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
DB_PATH = "users.db"


if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

if "resume" not in st.session_state:
    st.session_state.resume = ""

if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "report" not in st.session_state:
    st.session_state.report = ""

if "ats_score" not in st.session_state:
    st.session_state.ats_score = None

if "avg_score" not in st.session_state:
    st.session_state.avg_score = None

if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = ""

if "last_analysis_at" not in st.session_state:
    st.session_state.last_analysis_at = ""


st.title("AI Resume Analyzer")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34, 197, 94, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 30%),
            linear-gradient(180deg, #f8fffb 0%, #eef6ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1120px;
    }
    .hero-card {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .hero-copy {
        color: #475569;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .score-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        border: 1px solid rgba(34, 197, 94, 0.18);
        border-radius: 22px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 14px 30px rgba(34, 197, 94, 0.10);
    }
    .score-label {
        color: #166534;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .score-value {
        color: #052e16;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
    }
    .report-card {
        background: #0f172a;
        color: #e2e8f0;
        border-radius: 24px;
        padding: 1.35rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    }
    .report-card h1, .report-card h2, .report-card h3, .report-card p, .report-card li {
        color: #e2e8f0;
    }
    .auth-panel {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
    }
    .workspace-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .mini-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-top: 0.2rem;
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 1rem;
        box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">AI Resume Analyzer</div>
        <p class="hero-copy">
            Compare resumes against job descriptions, calculate semantic match scores, and generate
            targeted improvement suggestions with AI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                resume_filename TEXT,
                resume_text TEXT NOT NULL,
                job_description TEXT NOT NULL,
                ats_score REAL NOT NULL,
                avg_score REAL NOT NULL,
                report TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.commit()


def hash_password(password):
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return base64.b64encode(salt + pwd_hash).decode("utf-8")


def verify_password(password, stored_hash):
    decoded = base64.b64decode(stored_hash.encode("utf-8"))
    salt = decoded[:16]
    original_hash = decoded[16:]
    new_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return hmac.compare_digest(original_hash, new_hash)


def register_user(username, email, password):
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), hash_password(password)),
            )
            conn.commit()
        return True, "Registration successful. Please login."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def login_user(username, password):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()

    if row and verify_password(password, row[2]):
        return True, row[0], row[1]
    return False, None, None


@st.cache_resource
def load_ats_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as error:
        st.error(f"Error extracting text from PDF: {error}")
        return "Could not extract text from the PDF file."


def calculate_similarity_bert(text1, text2):
    ats_model = load_ats_model()
    embeddings1 = ats_model.encode([text1])
    embeddings2 = ats_model.encode([text2])
    return float(cosine_similarity(embeddings1, embeddings2)[0][0])


def get_report(resume, job_desc):
    if not api_key:
        return "Groq API key not found. Please add GROQ_API_KEY to your .env file."

    client = Groq(api_key=api_key)
    prompt = f"""
    # Context:
    - You are an AI Resume Analyzer, you will be given Candidate's resume and Job Description of the role he is applying for.

    # Instruction:
    - Analyze candidate's resume based on the possible points that can be extracted from job description, and give your evaluation on each point with the criteria below.
    - Consider all points like required skills, experience, etc that are needed for the job role.
    - Calculate the score to be given (out of 5) for every point based on evaluation at the beginning of each point with a detailed explanation.
    - If the resume aligns with the job description point, mark it with [MATCH] and provide a detailed explanation.
    - If the resume doesn't align with the job description point, mark it with [MISS] and provide a reason for it.
    - If a clear conclusion cannot be made, use [UNCLEAR] with a reason.
    - The final heading should be "Suggestions to improve your resume:" and give where and what the candidate can improve to be selected for that job role.

    # Inputs:
    Candidate Resume: {resume}
    ---
    Job Description: {job_desc}

    # Output:
    - Each and every point should be given a score (example: 3/5).
    - Mention the scores and status tag at the beginning of each point and then explain the reason.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content


def extract_scores(text):
    matches = re.findall(r"(\d+(?:\.\d+)?)/5", text)
    return [float(match) for match in matches]


def normalize_score(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytes):
        if len(value) == 8:
            return float(struct.unpack("d", value)[0])
        return float(value.decode("utf-8"))
    return float(value)


def save_analysis(user_id, resume_filename, resume_text, job_description, ats_score, avg_score, report):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                user_id, resume_filename, resume_text, job_description, ats_score, avg_score, report
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                resume_filename,
                resume_text,
                job_description,
                normalize_score(ats_score),
                normalize_score(avg_score),
                report,
            ),
        )
        conn.commit()


def get_latest_analysis(user_id):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT resume_filename, resume_text, job_description, ats_score, avg_score, report, created_at
            FROM analyses
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
    return row


def load_latest_analysis_into_session(user_id):
    latest_analysis = get_latest_analysis(user_id)
    if not latest_analysis:
        return False

    st.session_state.resume_filename = latest_analysis[0] or ""
    st.session_state.resume = latest_analysis[1]
    st.session_state.job_desc = latest_analysis[2]
    st.session_state.ats_score = normalize_score(latest_analysis[3])
    st.session_state.avg_score = normalize_score(latest_analysis[4])
    st.session_state.report = latest_analysis[5]
    st.session_state.last_analysis_at = latest_analysis[6] or ""
    st.session_state.form_submitted = True
    return True


def reset_analysis_state():
    st.session_state.form_submitted = False
    st.session_state.resume = ""
    st.session_state.job_desc = ""
    st.session_state.report = ""
    st.session_state.ats_score = None
    st.session_state.avg_score = None
    st.session_state.resume_filename = ""
    st.session_state.last_analysis_at = ""


def show_auth_page():
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="section-card">
                <h3 style="margin-top:0; color:#0f172a;">Welcome back</h3>
                <p class="hero-copy">
                    Create your account to keep the analyzer private and unlock a cleaner,
                    personalized experience before you start reviewing resumes.
                </p>
                <p class="mini-note">
                    Your credentials are stored locally in a secure hashed format inside <code>users.db</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h4 style="margin-top:0; color:#0f172a;">What you get</h4>
                <p class="mini-note">Semantic ATS-style similarity scoring</p>
                <p class="mini-note">AI-written point-by-point evaluation</p>
                <p class="mini-note">Downloadable report and repeat analysis flow</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown('<div class="auth-panel">', unsafe_allow_html=True)
        st.subheader("Login or create an account")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login")

                if login_submitted:
                    success, user_id, username = login_user(login_username, login_password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        reset_analysis_state()
                        load_latest_analysis_into_session(user_id)
                        st.success("Login successful.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with signup_tab:
            with st.form("signup_form"):
                signup_username = st.text_input("Choose a username")
                signup_email = st.text_input("Email")
                signup_password = st.text_input("Create a password", type="password")
                signup_confirm_password = st.text_input("Confirm password", type="password")
                signup_submitted = st.form_submit_button("Create account")

                if signup_submitted:
                    if not signup_username or not signup_email or not signup_password:
                        st.warning("Please fill all registration fields.")
                    elif signup_password != signup_confirm_password:
                        st.warning("Passwords do not match.")
                    elif len(signup_password) < 6:
                        st.warning("Password must be at least 6 characters long.")
                    else:
                        success, message = register_user(
                            signup_username, signup_email, signup_password
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)


def show_analyzer():
    st.sidebar.markdown("## Dashboard")
    st.sidebar.write(f"Signed in as `{st.session_state.username}`")
    st.sidebar.info("Upload a resume, paste a job description, and generate a detailed fit analysis.")

    header_col, action_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="workspace-head">
                    <div>
                        <h3 style="margin-top:0; color:#0f172a;">Resume Match Workspace</h3>
                        <p class="mini-note">Upload the candidate resume, paste the target job description, and let the analyzer score the fit.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.last_analysis_at:
            st.caption(
                f"Last saved analysis: {st.session_state.last_analysis_at}"
            )
    with action_col:
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_id = None
            reset_analysis_state()
            st.rerun()

    if not st.session_state.form_submitted:
        with st.form("my_form"):
            resume_file = st.file_uploader(
                label="Upload your Resume/CV in PDF format", type="pdf"
            )
            st.session_state.job_desc = st.text_area(
                "Enter the Job Description of the role you are applying for:",
                placeholder="Job Description...",
                value=st.session_state.job_desc,
            )

            submitted = st.form_submit_button("Analyze")
            if submitted:
                if st.session_state.job_desc and resume_file:
                    st.info("Extracting information")
                    st.session_state.resume_filename = resume_file.name
                    st.session_state.resume = extract_pdf_text(resume_file)
                    score_place = st.info("Generating scores...")
                    st.session_state.ats_score = calculate_similarity_bert(
                        st.session_state.resume, st.session_state.job_desc
                    )
                    st.session_state.report = get_report(
                        st.session_state.resume, st.session_state.job_desc
                    )
                    report_scores = extract_scores(st.session_state.report)
                    st.session_state.avg_score = (
                        float(sum(report_scores) / len(report_scores)) if report_scores else 0.0
                    )
                    save_analysis(
                        st.session_state.user_id,
                        st.session_state.resume_filename,
                        st.session_state.resume,
                        st.session_state.job_desc,
                        st.session_state.ats_score,
                        st.session_state.avg_score,
                        st.session_state.report,
                    )
                    load_latest_analysis_into_session(st.session_state.user_id)
                    score_place.success("Scores generated successfully.")
                    st.rerun()
                else:
                    st.warning("Please upload both Resume and Job Description to analyze.")

    if st.session_state.form_submitted:
        col1, col2 = st.columns(2, border=True)
        with col1:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">ATS Similarity Score</div>
                    <p class="score-value">{(st.session_state.ats_score or 0):.2%}</p>
                    <p class="mini-note">Semantic match between the resume content and the job description.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">Average AI Evaluation</div>
                    <p class="score-value">{(st.session_state.avg_score or 0):.2f}/5</p>
                    <p class="mini-note">Average extracted from the point-by-point LLM feedback.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.resume_filename:
            st.caption(f"Last uploaded resume: {st.session_state.resume_filename}")

        st.subheader("AI Generated Analysis Report")
        with st.container():
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(st.session_state.report)
            st.markdown("</div>", unsafe_allow_html=True)

        button_col1, button_col2 = st.columns(2)
        with button_col1:
            st.download_button(
                label="Download Report",
                data=st.session_state.report,
                file_name="report.txt",
            )
        with button_col2:
            if st.button("Analyze another resume", use_container_width=True):
                reset_analysis_state()
                st.rerun()


init_db()

if st.session_state.logged_in:
    show_analyzer()
else:
    show_auth_page()
