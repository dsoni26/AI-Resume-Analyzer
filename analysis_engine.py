import json
import re
from collections import Counter
from datetime import datetime
from statistics import mean


SECTION_ALIASES = {
    "skills": ["skills", "technical skills", "core competencies", "tech stack"],
    "experience": ["experience", "work experience", "professional experience", "employment"],
    "projects": ["projects", "project experience", "academic projects", "personal projects"],
    "education": ["education", "academic background", "qualifications"],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "using",
    "ability",
    "strong",
    "knowledge",
    "experience",
    "work",
    "working",
    "understanding",
    "plus",
    "good",
    "have",
    "has",
    "will",
    "your",
    "our",
    "their",
    "you",
}

KNOWN_SKILLS = {
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "go",
    "ruby",
    "php",
    "sql",
    "mysql",
    "postgresql",
    "sqlite",
    "mongodb",
    "redis",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "nlp",
    "machine learning",
    "deep learning",
    "data analysis",
    "data science",
    "data visualization",
    "streamlit",
    "flask",
    "django",
    "fastapi",
    "react",
    "next.js",
    "node.js",
    "html",
    "css",
    "tailwind css",
    "bootstrap",
    "rest api",
    "graphql",
    "docker",
    "docker compose",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "ci/cd",
    "github actions",
    "jenkins",
    "linux",
    "git",
    "github",
    "microservices",
    "oop",
    "problem solving",
    "communication",
    "leadership",
    "power bi",
    "tableau",
    "excel",
    "powerpoint",
    "word",
    "etl",
    "api integration",
    "unit testing",
    "pytest",
    "unittest",
    "agile",
    "scrum",
    "jira",
    "figma",
    "canva",
    "firebase",
    "supabase",
    "oauth",
    "authentication",
    "authorization",
    "resume screening",
    "ats",
    "sentence transformers",
    "cosine similarity",
    "reportlab",
    "fpdf",
    "pdfminer",
    "sqlite database",
    "database design",
    "data structures",
    "algorithms",
    "computer vision",
    "openai",
    "llm",
    "prompt engineering",
}

SKILL_PATTERNS = [
    re.compile(
        r"(?:experience|expertise|proficiency|knowledge|familiarity|hands-on experience)\s+(?:with|in)\s+([a-z0-9\+\#\/\-\.\s,]{3,100})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:skills|technologies|tools|stack|requirements)\s*[:\-]\s*([a-z0-9\+\#\/\-\.\s,]{3,200})",
        re.IGNORECASE,
    ),
]


def _slug(value):
    normalized = re.sub(r"[^a-z0-9\+\#]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _title_skill(skill):
    return " ".join(part.upper() if len(part) <= 3 and part.isalpha() else part.title() for part in skill.split())


def extract_skills(text):
    if not text:
        return []

    text_lower = text.lower()
    found = set()

    for skill in KNOWN_SKILLS:
        escaped = re.escape(skill)
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(_slug(skill))

    for pattern in SKILL_PATTERNS:
        for match in pattern.findall(text):
            for chunk in re.split(r",|/| and |\n|\|", match):
                candidate = _slug(chunk)
                if len(candidate) < 2 or candidate in STOPWORDS:
                    continue
                if 1 <= len(candidate.split()) <= 3:
                    found.add(candidate)

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\+\#\.\-/]{1,24}", text_lower)
    token_counts = Counter(_slug(token) for token in tokens if _slug(token) and _slug(token) not in STOPWORDS)
    for token, count in token_counts.items():
        if count >= 2 and token in {_slug(skill) for skill in KNOWN_SKILLS}:
            found.add(token)

    cleaned = sorted({skill for skill in found if skill and len(skill) > 1})
    return [_title_skill(skill) for skill in cleaned]


def extract_keywords(text, top_n=12):
    normalized = _slug(text)
    words = [
        word
        for word in normalized.split()
        if len(word) > 2 and word not in STOPWORDS and not word.isdigit()
    ]
    counts = Counter(words)
    return [word.title() for word, _ in counts.most_common(top_n)]


def split_sections(resume_text):
    if not resume_text:
        return {}

    lines = [line.strip() for line in resume_text.splitlines()]
    sections = {name: "" for name in SECTION_ALIASES}
    current = None
    buffer = []

    def flush():
        nonlocal buffer, current
        if current and buffer:
            sections[current] = (sections[current] + "\n" + "\n".join(buffer)).strip()
        buffer = []

    for line in lines:
        lowered = line.lower().strip(": ").strip()
        matched = None
        for section_name, aliases in SECTION_ALIASES.items():
            if lowered in aliases:
                matched = section_name
                break

        if matched:
            flush()
            current = matched
            continue

        if current and line:
            buffer.append(line)

    flush()
    return sections


def summarize_job_description(job_desc, job_skills):
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", job_desc).strip())
    summary_parts = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 40][:2]
    if job_skills:
        skills_line = ", ".join(job_skills[:6])
        summary_parts.append(f"Key focus areas include {skills_line}.")
    return " ".join(summary_parts) or "Job description summary unavailable."


def calculate_skill_gap(resume_skills, job_skills):
    resume_set = {skill.lower() for skill in resume_skills}
    job_set = {skill.lower() for skill in job_skills}
    matched = sorted(skill for skill in job_skills if skill.lower() in resume_set)
    missing = sorted(skill for skill in job_skills if skill.lower() not in resume_set)
    match_percentage = round((len(matched) / len(job_set) * 100), 1) if job_set else 0.0
    return matched, missing, match_percentage


def score_resume_sections(sections, matched_skills, missing_skills, job_desc):
    total_job_signals = max(len(matched_skills) + len(missing_skills), 1)
    skills_coverage = len(matched_skills) / total_job_signals

    def section_score(text, keywords=(), weight=1.0):
        if not text:
            return 1.0
        base = min(len(text.split()) / 90, 1.0) * 2.0
        hits = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        match_boost = min(hits / max(len(keywords), 1), 1.0) * 2.0
        return round(min(5.0, 1.0 + (base + match_boost) * weight), 1)

    scores = {
        "Skills": round(1.0 + skills_coverage * 4.0, 1),
        "Experience": section_score(sections.get("experience", ""), matched_skills[:8], 1.0),
        "Projects": section_score(sections.get("projects", ""), matched_skills[:6], 0.9),
        "Education": section_score(sections.get("education", ""), ["degree", "university", "bachelor", "master"], 0.7),
    }

    feedback = {
        "Skills": (
            "Strong alignment with the target stack."
            if scores["Skills"] >= 4
            else "Add a dedicated skills section with more role-specific keywords."
        ),
        "Experience": (
            "Experience section shows relevant signals for the role."
            if scores["Experience"] >= 4
            else "Quantify impact and align bullets with the job description's responsibilities."
        ),
        "Projects": (
            "Projects reinforce the target job requirements."
            if scores["Projects"] >= 4
            else "Add projects that demonstrate the missing tools, frameworks, or domain knowledge."
        ),
        "Education": (
            "Education section is clearly represented."
            if scores["Education"] >= 3.5
            else "Clarify degree details, coursework, or certifications to strengthen the academic profile."
        ),
    }

    if "certification" in job_desc.lower() and "certification" not in sections.get("education", "").lower():
        feedback["Education"] = "Mention relevant certifications if the role values them."

    return scores, feedback


def build_improvement_suggestions(missing_skills, section_scores, ats_score):
    suggestions = []
    if missing_skills:
        suggestions.append(
            f"Add evidence of these missing skills where applicable: {', '.join(missing_skills[:8])}."
        )
    if section_scores["Experience"] < 4:
        suggestions.append("Rewrite experience bullets using action verbs and measurable outcomes.")
    if section_scores["Projects"] < 4:
        suggestions.append("Highlight 1-2 projects that mirror the job's tech stack or business domain.")
    if section_scores["Skills"] < 4:
        suggestions.append("Create a concise keyword-rich skills block near the top of the resume.")
    if ats_score < 0.65:
        suggestions.append("Improve keyword alignment with the job description to lift ATS compatibility.")
    return suggestions[:5] or ["The resume is broadly aligned. Focus on adding measurable achievements."]


def build_keyword_suggestions(job_skills, missing_skills, resume_skills):
    high_priority = missing_skills[:6]
    supporting = [skill for skill in job_skills if skill not in high_priority and skill not in resume_skills][:4]
    return high_priority + supporting


def classify_strength(score):
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Strong"
    if score >= 50:
        return "Moderate"
    return "Needs Improvement"


def build_dynamic_feedback(ats_score, skill_match_percentage, strength_label):
    if strength_label == "Excellent":
        return "The resume is highly aligned and already reads like a strong shortlist candidate."
    if skill_match_percentage < 45:
        return "The biggest gap is keyword and skills coverage. Tightening that should unlock better ATS performance."
    if ats_score < 0.6:
        return "The resume has potential, but it needs stronger alignment with the target role's language and priorities."
    return "The resume is competitive, and a few targeted edits could make it interview-ready."


def create_fallback_ai_report(result):
    lines = [
        "## AI Evaluation",
        "",
        f"- ATS Score: **{result['resume_match_percentage']}%**",
        f"- Skill Match: **{result['skill_match_percentage']}%**",
        f"- Resume Strength: **{result['resume_strength_label']}**",
        "",
        "### Strengths",
    ]
    if result["matched_skills"]:
        for skill in result["matched_skills"][:6]:
            lines.append(f"- Demonstrates alignment with **{skill}**.")
    else:
        lines.append("- The resume would benefit from clearer job-specific skills.")

    lines.extend(["", "### Improvement Priorities"])
    for suggestion in result["improvement_suggestions"]:
        lines.append(f"- {suggestion}")

    return "\n".join(lines)


def build_report_markdown(result):
    matched = ", ".join(result["matched_skills"]) or "No matched skills detected yet."
    missing = ", ".join(result["missing_skills"]) or "No major missing skills detected."
    section_lines = "\n".join(
        f"- **{section}**: {score:.1f}/5 - {result['section_feedback'][section]}"
        for section, score in result["section_scores"].items()
    )
    improvement_lines = "\n".join(f"- {item}" for item in result["improvement_suggestions"])
    keyword_lines = "\n".join(f"- {item}" for item in result["keyword_optimization_suggestions"])

    return "\n".join(
        [
            "## Analysis Summary",
            "",
            f"- Candidate: **{result['candidate_name']}**",
            f"- ATS Score: **{result['resume_match_percentage']}%**",
            f"- Skill Match: **{result['skill_match_percentage']}%**",
            f"- Resume Strength: **{result['resume_strength_label']} ({result['resume_strength_score']}%)**",
            f"- Generated At: **{result['analysis_timestamp']}**",
            "",
            "### Job Description Summary",
            result["job_description_summary"],
            "",
            "### Matched Skills",
            matched,
            "",
            "### Missing Skills",
            missing,
            "",
            "### Section-wise Analysis",
            section_lines,
            "",
            "### Resume Improvement Suggestions",
            improvement_lines,
            "",
            "### Keyword Optimization Suggestions",
            keyword_lines,
            "",
            "### Dynamic Feedback",
            result["dynamic_feedback"],
            "",
            result["ai_report"],
        ]
    )


def build_analysis_result(username, resume_text, job_desc, ats_score, analysis_timestamp=None):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)
    if not job_skills:
        job_skills = extract_keywords(job_desc, top_n=10)

    matched_skills, missing_skills, skill_match_percentage = calculate_skill_gap(
        resume_skills, job_skills
    )
    sections = split_sections(resume_text)
    section_scores, section_feedback = score_resume_sections(
        sections, matched_skills, missing_skills, job_desc
    )
    avg_score = round(mean(section_scores.values()), 2)
    resume_strength_score = round(
        (ats_score * 100 * 0.35) + (skill_match_percentage * 0.35) + ((avg_score / 5) * 100 * 0.30),
        1,
    )
    result = {
        "version": 2,
        "candidate_name": username or "Candidate",
        "analysis_timestamp": analysis_timestamp
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resume_match_percentage": round(ats_score * 100, 1),
        "ats_score": round(float(ats_score), 4),
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_percentage": skill_match_percentage,
        "section_scores": section_scores,
        "section_feedback": section_feedback,
        "keyword_optimization_suggestions": build_keyword_suggestions(
            job_skills, missing_skills, resume_skills
        ),
        "improvement_suggestions": build_improvement_suggestions(
            missing_skills, section_scores, float(ats_score)
        ),
        "job_description_summary": summarize_job_description(job_desc, job_skills),
        "resume_strength_score": resume_strength_score,
        "resume_strength_label": classify_strength(resume_strength_score),
        "dynamic_feedback": "",
        "ai_report": "",
        "report_markdown": "",
        "avg_score": avg_score,
    }
    result["dynamic_feedback"] = build_dynamic_feedback(
        float(ats_score), skill_match_percentage, result["resume_strength_label"]
    )
    result["ai_report"] = create_fallback_ai_report(result)
    result["report_markdown"] = build_report_markdown(result)
    return result


def update_result_with_ai_report(result, ai_report):
    result["ai_report"] = ai_report.strip() if ai_report else create_fallback_ai_report(result)
    result["report_markdown"] = build_report_markdown(result)
    return result


def serialize_analysis_result(result):
    return json.dumps(result, ensure_ascii=True)


def parse_saved_report(raw_report, username, resume_text, job_desc, ats_score, avg_score, created_at):
    if not raw_report:
        result = build_analysis_result(username, resume_text, job_desc, ats_score, created_at)
        result["avg_score"] = avg_score or result["avg_score"]
        return result

    try:
        result = json.loads(raw_report)
        if isinstance(result, dict) and result.get("version") == 2:
            return result
    except json.JSONDecodeError:
        pass

    result = build_analysis_result(username, resume_text, job_desc, ats_score, created_at)
    result["ai_report"] = raw_report
    result["avg_score"] = avg_score or result["avg_score"]
    result["report_markdown"] = build_report_markdown(result)
    return result
