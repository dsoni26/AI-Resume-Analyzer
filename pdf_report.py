from io import BytesIO

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _meter(score, width=420, height=20, label="Score"):
    drawing = Drawing(width, height + 16)
    drawing.add(Rect(0, 8, width, height, fillColor=colors.HexColor("#E2E8F0"), strokeColor=None))
    drawing.add(
        Rect(
            0,
            8,
            max(0, min(width, width * (score / 100.0))),
            height,
            fillColor=colors.HexColor("#0F766E"),
            strokeColor=None,
        )
    )
    drawing.add(
        String(
            0,
            0,
            f"{label}: {score:.1f}%",
            fontName="Helvetica-Bold",
            fontSize=9,
            fillColor=colors.HexColor("#0F172A"),
        )
    )
    return drawing


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CardTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#475569"),
            alignment=TA_LEFT,
        )
    )
    return styles


def _key_value_table(result):
    rows = [
        ["Candidate", result["candidate_name"]],
        ["ATS Score", f"{result['resume_match_percentage']}%"],
        ["Skill Match", f"{result['skill_match_percentage']}%"],
        ["Resume Strength", f"{result['resume_strength_label']} ({result['resume_strength_score']}%)"],
        ["Average Section Score", f"{result['avg_score']:.2f}/5"],
        ["Analysis Timestamp", result["analysis_timestamp"]],
    ]
    table = Table(rows, colWidths=[1.8 * inch, 4.7 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _skills_table(title, skills, background):
    values = [[title]]
    if skills:
        for skill in skills:
            values.append([skill])
    else:
        values.append(["None identified"])

    table = Table(values, colWidths=[3.1 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), background),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _section_scores_table(result):
    rows = [["Section", "Score", "Feedback"]]
    for section, score in result["section_scores"].items():
        rows.append([section, f"{score:.1f}/5", result["section_feedback"][section]])

    table = Table(rows, colWidths=[1.3 * inch, 1.0 * inch, 4.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def build_pdf_report(result):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=30,
    )
    styles = _build_styles()
    story = []

    story.append(Paragraph("AI Resume Analyzer Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Professional ATS and skill-gap evaluation report", styles["BodyMuted"]))
    story.append(Spacer(1, 14))
    story.append(_key_value_table(result))
    story.append(Spacer(1, 14))
    story.append(_meter(result["resume_strength_score"], label="Resume Strength Meter"))
    story.append(Spacer(1, 10))
    story.append(_meter(result["resume_match_percentage"], label="Resume Match Percentage"))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Job Description Summary", styles["CardTitle"]))
    story.append(Paragraph(result["job_description_summary"], styles["BodyText"]))
    story.append(Spacer(1, 14))

    skill_tables = Table(
        [
            [
                _skills_table("Matched Skills", result["matched_skills"], colors.HexColor("#15803D")),
                _skills_table("Missing Skills", result["missing_skills"], colors.HexColor("#B91C1C")),
            ]
        ],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    story.append(skill_tables)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Section-wise Analysis", styles["CardTitle"]))
    story.append(_section_scores_table(result))
    story.append(Spacer(1, 16))

    story.append(Paragraph("Resume Improvement Suggestions", styles["CardTitle"]))
    for suggestion in result["improvement_suggestions"]:
        story.append(Paragraph(f"• {suggestion}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Keyword Optimization Suggestions", styles["CardTitle"]))
    for suggestion in result["keyword_optimization_suggestions"]:
        story.append(Paragraph(f"• {suggestion}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Dynamic Feedback", styles["CardTitle"]))
    story.append(Paragraph(result["dynamic_feedback"], styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI Summary", styles["CardTitle"]))
    story.append(Paragraph(result["ai_report"].replace("\n", "<br/>"), styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()
