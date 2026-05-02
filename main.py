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
    return


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
        st.caption("None identified")
        return
    for item in items:
        if tone == "success":
            st.success(item, icon="✅")
        elif tone == "danger":
            st.warning(item, icon="⚠️")
        else:
            st.write(item)


def render_footer():
    st.divider()
    st.caption(
        "Portfolio-ready build: AI Resume Analyzer with ATS scoring, skill-gap analysis, PDF reporting, Docker deployment, and CI/CD support."
    )


def show_auth_page():
    left_col, right_col = st.columns([1.08, 0.92], gap="large")

    with left_col:
        st.caption("SMART RESUME SCREENING")
        st.title("Build interview-ready resumes with clearer ATS insights.")
        st.write(
            "Compare resumes against role requirements, uncover missing skills, and generate a professional recruiter-friendly analysis report in seconds."
        )
        st.subheader("What this dashboard includes")
        st.write("- Semantic ATS scoring with Sentence Transformers")
        st.write("- Missing skills detection and section-wise resume scoring")
        st.write("- Professional downloadable PDF report for portfolio-ready demos")
        st.write("- Secure local authentication with hashed passwords in SQLite")

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
        st.subheader("Resume Strength Meter")
        st.metric("Strength", result["resume_strength_label"])
        st.info(result["dynamic_feedback"])
        st.progress(min(max(result["resume_strength_score"] / 100, 0.0), 1.0))
        st.caption(f"Analysis timestamp: {result['analysis_timestamp']}")
        st.caption(f"Last uploaded resume: {st.session_state.resume_filename}")

    with col2:
        st.subheader("Job Description Summary")
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

    st.title("Professional AI Resume Analysis Workspace")
    st.write(
        "Upload a resume, compare it against a target job description, and generate ATS, skill-gap, section-wise, and recruiter-facing report insights."
    )

    if st.session_state.active_view == "Profile":
        st.subheader("User Profile")
        st.write("Your authentication and saved analysis history remain stored locally in SQLite.")
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
