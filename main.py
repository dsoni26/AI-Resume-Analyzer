import base64
import hashlib
import hmac
import os
import sqlite3
import struct

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from analysis_engine import (
    build_analysis_result,
    create_fallback_ai_report,
    parse_saved_report,
    serialize_analysis_result,
    update_result_with_ai_report,
)
from pdf_report import build_pdf_report


load_dotenv()

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

api_key = os.getenv("GROQ_API_KEY")
DB_PATH = "users.db"

DEFAULT_SESSION_STATE = {
    "form_submitted": False,
    "resume": "",
    "job_desc": "",
    "logged_in": False,
    "username": "",
    "user_id": None,
    "report": "",
    "ats_score": None,
    "avg_score": None,
    "resume_filename": "",
    "last_analysis_at": "",
    "analysis_result": None,
    "active_view": "Analyzer",
}


def initialize_session_state():
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 30%),
                linear-gradient(180deg, #f8fafc 0%, #edf4ff 100%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #0f172a;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0;
        }
        .hero-card, .surface-card, .insight-card, .upload-card, .auth-panel, .footer-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 24px;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
        }
        .hero-card {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1rem;
            background: linear-gradient(140deg, rgba(255,255,255,0.95), rgba(240,249,255,0.95));
        }
        .surface-card, .upload-card, .auth-panel, .footer-card {
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
        }
        .insight-card {
            padding: 1rem 1.1rem;
            min-height: 148px;
        }
        .eyebrow {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            color: #0f172a;
            font-size: 2.15rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
        }
        .hero-copy, .muted-copy {
            color: #475569;
            line-height: 1.6;
            margin: 0;
        }
        .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.5rem;
        }
        .tag {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .tag-success {
            color: #166534;
            background: #dcfce7;
            border-color: #86efac;
        }
        .tag-danger {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fca5a5;
        }
        .tag-neutral {
            color: #0f172a;
            background: #e2e8f0;
            border-color: #cbd5e1;
        }
        .footer-card {
            margin-top: 1rem;
            text-align: center;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
        }
        div[data-testid="stForm"] {
            background: transparent;
            border: none;
            padding: 0;
            box-shadow: none;
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            background: rgba(255,255,255,0.88);
        }
        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            font-weight: 700;
            color: #0f172a;
            background: #dbeafe;
        }
        </style>
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
    return SentenceTransformer("all-MiniLM-L6-v2")


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


def get_ai_report(resume, job_desc, analysis_result):
    if not api_key:
        return create_fallback_ai_report(analysis_result)

    client = Groq(api_key=api_key)
    prompt = f"""
You are an expert resume reviewer.

Write a concise, recruiter-friendly markdown report with these sections only:
1. Executive Summary
2. Strengths
3. Missing Skills and Gaps
4. Resume Improvement Suggestions

Use the structured inputs below instead of repeating raw content.

Candidate: {analysis_result['candidate_name']}
ATS Score: {analysis_result['resume_match_percentage']}%
Skill Match: {analysis_result['skill_match_percentage']}%
Matched Skills: {", ".join(analysis_result['matched_skills'])}
Missing Skills: {", ".join(analysis_result['missing_skills'])}
Section Scores: {analysis_result['section_scores']}
Dynamic Feedback: {analysis_result['dynamic_feedback']}
Job Summary: {analysis_result['job_description_summary']}

Resume:
{resume[:5000]}

Job Description:
{job_desc[:5000]}
"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception:
        return create_fallback_ai_report(analysis_result)


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
    st.session_state.analysis_result = parse_saved_report(
        latest_analysis[5],
        st.session_state.username,
        st.session_state.resume,
        st.session_state.job_desc,
        st.session_state.ats_score,
        st.session_state.avg_score,
        latest_analysis[6] or "",
    )
    st.session_state.report = st.session_state.analysis_result["report_markdown"]
    st.session_state.last_analysis_at = latest_analysis[6] or ""
    st.session_state.form_submitted = True
    return True


def reset_analysis_state():
    for key in [
        "form_submitted",
        "resume",
        "job_desc",
        "report",
        "ats_score",
        "avg_score",
        "resume_filename",
        "last_analysis_at",
        "analysis_result",
    ]:
        st.session_state[key] = DEFAULT_SESSION_STATE[key]


def render_tag_group(title, items, tone):
    st.markdown(f"**{title}**")
    if not items:
        st.markdown('<div class="tag-row"><span class="tag tag-neutral">None identified</span></div>', unsafe_allow_html=True)
        return
    tags = "".join(
        f'<span class="tag tag-{tone}">{item}</span>'
        for item in items
    )
    st.markdown(f'<div class="tag-row">{tags}</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        """
        <div class="footer-card">
            <div class="eyebrow">Portfolio-Ready Build</div>
            <p class="muted-copy">
                AI Resume Analyzer with ATS scoring, skill-gap analysis, structured feedback,
                PDF reporting, Docker deployment, and CI/CD workflow support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_auth_page():
    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="hero-card">
                <div class="eyebrow">Smart Resume Screening</div>
                <div class="hero-title">Build interview-ready resumes with clearer ATS insights.</div>
                <p class="hero-copy">
                    Compare resumes against role requirements, uncover missing skills, and generate
                    a professional recruiter-friendly analysis report in seconds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="surface-card">
                <h3 style="margin-top:0;">What this dashboard includes</h3>
                <p class="muted-copy">Semantic ATS scoring with Sentence Transformers</p>
                <p class="muted-copy">Missing skills detection and section-wise resume scoring</p>
                <p class="muted-copy">Professional downloadable PDF report for portfolio-ready demos</p>
                <p class="muted-copy">Secure local authentication with hashed passwords in SQLite</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.subheader("Login or Create an Account")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input("Password", type="password")
                login_submitted = st.form_submit_button("Login", use_container_width=True)

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
                signup_submitted = st.form_submit_button("Create account", use_container_width=True)

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
    render_footer()


def render_sidebar():
    with st.sidebar:
        st.markdown("## Resume Analyzer")
        st.write(f"Signed in as `{st.session_state.username}`")
        st.caption("AI + ATS + PDF reporting dashboard")
        st.session_state.active_view = st.radio(
            "Navigation",
            ["Analyzer", "Latest Report", "Profile"],
            index=["Analyzer", "Latest Report", "Profile"].index(st.session_state.active_view),
        )
        st.divider()
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            st.metric("ATS Score", f"{result['resume_match_percentage']}%")
            st.metric("Skill Match", f"{result['skill_match_percentage']}%")
            st.metric("Resume Strength", f"{result['resume_strength_score']}%")
            st.progress(min(max(result["resume_strength_score"] / 100, 0.0), 1.0))
        else:
            st.info("Run an analysis to unlock the dashboard insights.")
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_id = None
            reset_analysis_state()
            st.rerun()


def analyze_resume_flow(resume_file, job_desc):
    with st.spinner("Extracting resume content and building the analysis dashboard..."):
        resume_text = extract_pdf_text(resume_file)
        ats_score = calculate_similarity_bert(resume_text, job_desc)
        analysis_result = build_analysis_result(
            st.session_state.username,
            resume_text,
            job_desc,
            ats_score,
        )
        ai_report = get_ai_report(resume_text, job_desc, analysis_result)
        analysis_result = update_result_with_ai_report(analysis_result, ai_report)

    st.session_state.resume_filename = resume_file.name
    st.session_state.resume = resume_text
    st.session_state.job_desc = job_desc
    st.session_state.ats_score = ats_score
    st.session_state.avg_score = analysis_result["avg_score"]
    st.session_state.analysis_result = analysis_result
    st.session_state.report = analysis_result["report_markdown"]
    st.session_state.form_submitted = True

    save_analysis(
        st.session_state.user_id,
        st.session_state.resume_filename,
        st.session_state.resume,
        st.session_state.job_desc,
        st.session_state.ats_score,
        st.session_state.avg_score,
        serialize_analysis_result(analysis_result),
    )
    load_latest_analysis_into_session(st.session_state.user_id)


def render_overview_cards(result):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ATS Score", f"{result['resume_match_percentage']}%")
    col2.metric("Skill Match", f"{result['skill_match_percentage']}%")
    col3.metric("Average Section Score", f"{result['avg_score']:.2f}/5")
    col4.metric("Resume Strength", f"{result['resume_strength_score']}%")


def render_strength_section(result):
    col1, col2 = st.columns([1.05, 0.95], gap="large")
    with col1:
        st.markdown(
            f"""
            <div class="surface-card">
                <div class="eyebrow">Resume Strength Meter</div>
                <h3 style="margin-top:0;">{result['resume_strength_label']}</h3>
                <span class="status-pill">{result['dynamic_feedback']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(result["resume_strength_score"] / 100, 0.0), 1.0))
        st.caption(f"Analysis timestamp: {result['analysis_timestamp']}")
        st.caption(f"Last uploaded resume: {st.session_state.resume_filename}")

    with col2:
        st.markdown(
            """
            <div class="eyebrow">Role Snapshot</div>
            <h3 style="margin-top:0;">Job Description Summary</h3>
            """,
            unsafe_allow_html=True,
        )
        st.write(result["job_description_summary"])


def render_analysis_dashboard(result):
    render_overview_cards(result)
    st.markdown("")
    render_strength_section(result)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Skill Gap", "Section Analysis", "AI Report"]
    )

    with tab1:
        insight1, insight2 = st.columns(2, gap="large")
        with insight1:
            st.markdown("### Improvement Suggestions")
            for suggestion in result["improvement_suggestions"]:
                st.write(f"• {suggestion}")

        with insight2:
            st.markdown("### Keyword Optimization")
            for keyword in result["keyword_optimization_suggestions"]:
                st.write(f"• {keyword}")

        with st.expander("View saved analysis markdown"):
            st.markdown(result["report_markdown"])

    with tab2:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            render_tag_group("Matched Skills", result["matched_skills"], "success")
        with col2:
            render_tag_group("Missing Skills", result["missing_skills"], "danger")
        st.markdown("")
        st.progress(min(max(result["skill_match_percentage"] / 100, 0.0), 1.0))
        st.caption(f"Skill match percentage: {result['skill_match_percentage']}%")
        if result["missing_skills"]:
            st.info(
                "Recommendation: reflect the missing skills in your experience, projects, or certifications section where genuinely applicable."
            )
        else:
            st.success("Great alignment. No major missing skills were detected against the target role.")

    with tab3:
        score_cols = st.columns(len(result["section_scores"]))
        for index, (section, score) in enumerate(result["section_scores"].items()):
            with score_cols[index]:
                st.metric(section, f"{score:.1f}/5")
                st.progress(min(max(score / 5, 0.0), 1.0))

        for section, feedback in result["section_feedback"].items():
            with st.expander(f"{section} feedback"):
                st.write(feedback)

    with tab4:
        st.markdown(result["ai_report"])

    pdf_bytes = build_pdf_report(result)
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        st.download_button(
            label="Download Analysis Report (PDF)",
            data=pdf_bytes,
            file_name="analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with action_col2:
        if st.button("Analyze Another Resume", use_container_width=True):
            reset_analysis_state()
            st.rerun()


def show_analyzer():
    render_sidebar()

    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Dashboard</div>
            <div class="hero-title">Professional AI resume analysis workspace</div>
            <p class="hero-copy">
                Upload a resume, compare it against a target job description, and generate
                ATS, skill-gap, section-wise, and recruiter-facing report insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.active_view == "Profile":
        st.markdown(
            """
            <div class="surface-card">
                <div class="eyebrow">Account Snapshot</div>
                <h3 style="margin-top:0;">User Profile</h3>
                <p class="muted-copy">Your authentication and saved analysis history remain stored locally in SQLite.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("User", st.session_state.username)
        st.metric("Latest Analysis Saved", st.session_state.last_analysis_at or "No saved analysis yet")
        render_footer()
        return

    if st.session_state.active_view == "Latest Report" and not st.session_state.analysis_result:
        st.info("No saved analysis found yet. Run your first resume analysis from the Analyzer tab.")

    if st.session_state.active_view == "Analyzer":
        st.subheader("Upload Resume and Target Job Description")
        with st.form("analysis_form"):
            resume_file = st.file_uploader(
                label="Upload your Resume/CV in PDF format",
                type="pdf",
                help="PDF only. The analyzer extracts text and compares it against the job description.",
            )
            job_desc = st.text_area(
                "Enter the Job Description",
                placeholder="Paste the role requirements, responsibilities, and preferred qualifications here...",
                value=st.session_state.job_desc,
                height=220,
            )
            submitted = st.form_submit_button("Run Advanced Analysis", use_container_width=True)

        if submitted:
            if job_desc and resume_file:
                analyze_resume_flow(resume_file, job_desc)
                st.success("Analysis completed successfully.")
                st.rerun()
            else:
                st.warning("Please upload both a resume PDF and a job description.")

    if st.session_state.analysis_result and st.session_state.active_view in {"Analyzer", "Latest Report"}:
        render_analysis_dashboard(st.session_state.analysis_result)

    render_footer()


initialize_session_state()
inject_styles()
init_db()

if st.session_state.logged_in:
    show_analyzer()
else:
    show_auth_page()
