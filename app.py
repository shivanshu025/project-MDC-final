"""
TalentRank AI — v3.0  (Upgraded)
─────────────────────────────────────────────────────────────────────────────
AI Engine:   sentence-transformers/all-MiniLM-L6-v2  (Transformer Embeddings)
             Hybrid scored: Semantic Similarity + Skill Match + Exp + Edu
No paid API. No GPU required. Fully offline after model download.

Run:   python app.py
Open:  http://localhost:5000
─────────────────────────────────────────────────────────────────────────────
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import io
import json
import csv
import os
from datetime import datetime

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("./local_model")
    TRANSFORMER_SUPPORT = True
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    TRANSFORMER_SUPPORT = False

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import mm, inch
    REPORTLAB_SUPPORT = True
except ImportError:
    REPORTLAB_SUPPORT = False

try:
    import pandas as pd
    PANDAS_SUPPORT = True
except ImportError:
    PANDAS_SUPPORT = False

app = Flask(__name__, static_folder="static")

# ─────────────────────────────────────────────────────────────────────────────
#  SKILL DICTIONARY  (120+ skills)
# ─────────────────────────────────────────────────────────────────────────────
SKILL_DICT = [
    # Languages
    "python","javascript","typescript","java","golang","go","rust","c++","c#","kotlin",
    "swift","php","ruby","scala","r","matlab","bash","shell","sql","perl","haskell",
    # Frontend
    "react","vue","angular","svelte","next.js","nuxt","html","css","tailwind","sass",
    "webpack","vite","redux","mobx","jquery","bootstrap",
    # Backend
    "node.js","django","flask","fastapi","spring","rails","express","nestjs",
    "graphql","rest api","grpc","websocket","microservices","serverless",
    # Data / AI / ML
    "machine learning","deep learning","nlp","computer vision",
    "pytorch","tensorflow","keras","scikit-learn",
    "openai","anthropic","gemini","llm","rag","langchain","hugging face",
    "transformers","gpt","bert","embeddings","vector database",
    "pandas","numpy","scipy","matplotlib","seaborn","plotly",
    "spark","airflow","kafka","dbt","etl","data pipeline",
    "pgvector","pinecone","weaviate","chroma",
    "sentence transformers","xgboost","lightgbm",
    # Databases
    "postgresql","mysql","mongodb","redis","sqlite","dynamodb",
    "firebase","supabase","elasticsearch","cassandra","neo4j",
    # Cloud / Infra / DevOps
    "aws","azure","gcp","google cloud","docker","kubernetes","terraform",
    "ci/cd","devops","linux","lambda","s3","ec2","ansible","cloudformation",
    "jenkins","gitlab","github actions",
    # Tools / Process
    "git","github","agile","scrum","jira","product management","startup",
    "react native","flutter","ios","android","mobile",
    # Security / Other
    "cybersecurity","penetration testing","blockchain","solidity","web3",
]

_CAPS = {"llm","nlp","aws","gcp","api","sql","css","html","gpu","rest",
         "grpc","etl","rag","ios","mit","cs","ux","ui","s3","ec2","php"}
_SPECIAL = {
    "node.js":"Node.js","next.js":"Next.js","postgresql":"PostgreSQL",
    "ci/cd":"CI/CD","scikit-learn":"scikit-learn","google cloud":"Google Cloud",
    "react native":"React Native","machine learning":"Machine Learning",
    "deep learning":"Deep Learning","computer vision":"Computer Vision",
    "vector database":"Vector Database","data pipeline":"Data Pipeline",
    "hugging face":"Hugging Face","openai":"OpenAI","anthropic":"Anthropic",
    "pgvector":"pgvector","typescript":"TypeScript","javascript":"JavaScript",
    "react":"React","vue":"Vue","python":"Python","golang":"Go","rust":"Rust",
    "c++":"C++","c#":"C#","kotlin":"Kotlin","swift":"Swift","php":"PHP",
    "ruby":"Ruby","scala":"Scala","angular":"Angular","svelte":"Svelte",
    "django":"Django","flask":"Flask","fastapi":"FastAPI","redis":"Redis",
    "mongodb":"MongoDB","firebase":"Firebase","supabase":"Supabase",
    "docker":"Docker","kubernetes":"Kubernetes","terraform":"Terraform",
    "github":"GitHub","linux":"Linux","pandas":"Pandas","numpy":"NumPy",
    "pytorch":"PyTorch","tensorflow":"TensorFlow","keras":"Keras",
    "spark":"Apache Spark","kafka":"Kafka","airflow":"Apache Airflow",
    "pinecone":"Pinecone","weaviate":"Weaviate","langchain":"LangChain",
    "gemini":"Gemini","gpt":"GPT","bert":"BERT","aws":"AWS","azure":"Azure",
    "gcp":"GCP","grpc":"gRPC","graphql":"GraphQL","rest api":"REST API",
    "websocket":"WebSocket","microservices":"Microservices","serverless":"Serverless",
    "flutter":"Flutter","embeddings":"Embeddings","chroma":"Chroma","dbt":"dbt",
    "sentence transformers":"Sentence Transformers","xgboost":"XGBoost",
    "lightgbm":"LightGBM","jira":"Jira","github actions":"GitHub Actions",
    "bootstrap":"Bootstrap","tailwind":"Tailwind","jquery":"jQuery",
    "cassandra":"Cassandra","neo4j":"Neo4j","solidity":"Solidity",
    "blockchain":"Blockchain","mobile":"Mobile",
}

def fmt_skill(s: str) -> str:
    if s in _SPECIAL: return _SPECIAL[s]
    if s in _CAPS:    return s.upper()
    return s.title()

# ─────────────────────────────────────────────────────────────────────────────
#  TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def extract_skills(text: str) -> list:
    tl = text.lower()
    seen, found = set(), []
    for skill in SKILL_DICT:
        if skill in seen: continue
        if re.search(r'\b' + re.escape(skill) + r'\b', tl):
            seen.add(skill)
            found.append(skill)
    return found

def extract_years(text: str) -> int:
    # First try explicit "5 years" / "5+ years"
    m = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', text, re.I)
    if m:
        return max(int(x) for x in m)
    # Calculate from date ranges — handles "2019–Present", "2019-2024", "2019 to 2024"
    current_year = 2026
    present = re.search(r'\b(20\d{2}|19\d{2})\b.*?(?:present|current|now)', text, re.I)
    if present:
        start = int(present.group(1))
        return current_year - start
    # fallback: find all 4-digit years, take range
    years_found = [int(y) for y in re.findall(r'\b(20\d{2}|19\d{2})\b', text)]
    if len(years_found) >= 2:
        return current_year - min(years_found)
    return 0

def detect_seniority(text: str):
    t = text.lower()
    if re.search(r'\b(staff|principal|distinguished|vp\b|cto\b|1[0-9]\+?\s*years?)', t): return "Staff", 4
    if re.search(r'\b(senior|sr\.?\b|lead|[7-9]\+?\s*years?)', t):    return "Senior", 3
    if re.search(r'\b(mid.?level|intermediate|[4-6]\+?\s*years?)', t): return "Mid", 2
    if re.search(r'\b(junior|jr\.?\b|entry.?level|associate|[0-3]\+?\s*years?)', t): return "Junior", 1
    # No keyword found — infer from years
    yrs = extract_years(text)
    if yrs >= 10: return "Staff",  4
    if yrs >= 6:  return "Senior", 3
    if yrs >= 3:  return "Mid",    2
    return "Junior", 1

# ─────────────────────────────────────────────────────────────────────────────
#  AI ENGINE — Transformer Embeddings (with TF-IDF fallback)
# ─────────────────────────────────────────────────────────────────────────────
def semantic_similarity(doc_a: str, doc_b: str) -> float:
    """Compute semantic cosine similarity between two texts."""
    if TRANSFORMER_SUPPORT:
        try:
            emb = _MODEL.encode([doc_a, doc_b], convert_to_numpy=True)
            sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
            return float(np.clip(sim, 0, 1))
        except Exception:
            pass
    # Fallback: TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cs
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3),
                                  min_df=1, max_features=8000, sublinear_tf=True)
    try:
        mat = vectorizer.fit_transform([doc_a, doc_b])
        return float(cs(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  HYBRID SCORING ENGINE
#  Final AI Score = 60% Semantic + 25% Skill + 10% Experience + 5% Education
# ─────────────────────────────────────────────────────────────────────────────
def compute_skill_score(jd_skills: list, resume_skills: list) -> int:
    jd_set  = set(jd_skills)
    res_set = set(resume_skills)
    if not jd_set: return 70
    overlap = len(jd_set & res_set) / len(jd_set)
    return min(100, int(overlap * 100))

def compute_experience_score(jd: str, resume: str) -> int:
    jd_yrs  = extract_years(jd) or 3
    res_yrs = extract_years(resume)
    _, jd_lvl  = detect_seniority(jd)
    _, res_lvl = detect_seniority(resume)

    # If seniority keywords missing, infer level from years
    if res_lvl == 2 and res_yrs >= 7:   res_lvl = 3  # treat as Senior
    if res_lvl == 2 and res_yrs >= 10:  res_lvl = 4  # treat as Staff

    yrs_sc = min(100, int((res_yrs / max(jd_yrs, 1)) * 100))
    lvl_sc = 100 if res_lvl >= jd_lvl else int((res_lvl / max(jd_lvl, 1)) * 100)
    bonus = 0
    jl, rl = jd.lower(), resume.lower()
    if 'startup' in jl and 'startup' in rl: bonus += 8
    if 'product' in jl and 'product' in rl: bonus += 4
    return min(100, int(yrs_sc * 0.5 + lvl_sc * 0.5) + bonus)

def compute_education_score(resume: str) -> int:
    r = resume.lower()
    if re.search(r'\b(ph\.?d|doctorate)\b', r):                     return 100
    if re.search(r'\b(m\.?s|m\.?eng|m\.?tech|masters?|mba)\b', r): return 90
    if re.search(r'\b(b\.?s|b\.?eng|b\.?tech|bachelors?)\b', r):   return 80
    if re.search(r'\b(associate|diploma|bootcamp)\b', r):           return 65
    if re.search(r'\b(self.?taught|autodidact)\b', r):              return 62
    return 68

def compute_final_score(semantic_sim: float, skill_sc: int, exp_sc: int, edu_sc: int) -> int:
    """Weights: 60% Semantic + 25% Skill + 10% Experience + 5% Education = 100%"""
    score = (semantic_sim * 100 * 0.60) + (skill_sc * 0.25) + (exp_sc * 0.10) + (edu_sc * 0.05)
    return min(100, int(score))

def get_recommendation(score: int) -> dict:
    if score >= 80:   return {"label": "Recommended",     "class": "rec-green",  "emoji": "✅"}
    if score >= 65:   return {"label": "Consider",        "class": "rec-yellow", "emoji": "🔍"}
    return               {"label": "Not Recommended", "class": "rec-red",    "emoji": "❌"}

# ─────────────────────────────────────────────────────────────────────────────
#  NARRATIVE GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def gen_ai_analysis(name: str, score: int, matched: list, missing: list,
                    semantic: float, yrs: int, resume: str, jd: str) -> dict:
    first = name.split()[0] if name else "This candidate"
    rec   = get_recommendation(score)

    # Strengths
    strengths = []
    if yrs >= 5:  strengths.append(f"{yrs}+ years of hands-on experience")
    elif yrs > 0: strengths.append(f"{yrs} years of relevant experience")
    top = [fmt_skill(s) for s in matched[:3]]
    if top: strengths.append(f"Strong in: {', '.join(top)}")
    r, j = resume.lower(), jd.lower()
    if re.search(r'\b(founder|cto|startup|0→1)\b', r):
        strengths.append("Startup & 0-to-1 building experience")
    if re.search(r'\b(lead|architect|principal|staff)\b', r):
        strengths.append("Technical leadership experience")
    if re.search(r'\b(shipped|launched|deployed|production)\b', r):
        strengths.append("Track record of shipping to production")
    if ('llm' in j or 'ai' in j) and re.search(r'\b(openai|anthropic|llm|gpt|claude)\b', r):
        strengths.append("LLM / AI integration experience")
    if len(strengths) < 2: strengths.append("Broad engineering skill set")

    # Improvement areas
    improvements = [fmt_skill(s) for s in missing[:5]]

    # Summary
    if score >= 80:
        summary = f"Strong semantic alignment detected. {first} is an excellent fit for this role with {len(matched)} matching skills."
    elif score >= 65:
        summary = f"{first} shows good alignment with {len(matched)} matched skills. Some gaps exist but the candidate is worth considering."
    elif score >= 55:
        summary = f"{first} has partial overlap with the job requirements. Upskilling in {len(missing)} areas recommended."
    else:
        summary = f"{first} currently lacks several key requirements for this role."

    return {
        "summary":      summary,
        "strengths":    strengths[:4],
        "improvements": improvements,
        "recommendation_label": rec["label"],
    }

# ─────────────────────────────────────────────────────────────────────────────
#  ADDITIONAL SKILLS (resume has but JD doesn't require)
# ─────────────────────────────────────────────────────────────────────────────
def get_additional_skills(jd_skills: list, resume_skills: list) -> list:
    jd_set = set(jd_skills)
    return [s for s in resume_skills if s not in jd_set]

# ─────────────────────────────────────────────────────────────────────────────
#  PDF REPORT GENERATION  (ReportLab)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(candidate: dict, rank: int, total: int) -> bytes:
    if not REPORTLAB_SUPPORT:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=20*mm, rightMargin=20*mm,
                             topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    story  = []

    # Colors
    PURPLE  = colors.HexColor("#7c6af7")
    DARK    = colors.HexColor("#07080e")
    GREEN   = colors.HexColor("#10b981")
    RED     = colors.HexColor("#f87171")
    ORANGE  = colors.HexColor("#fb923c")
    LIGHT   = colors.HexColor("#f8fafc")
    MUTED   = colors.HexColor("#64748b")
    WHITE   = colors.white

    # Custom styles
    title_style = ParagraphStyle("TR_Title", parent=styles["Title"],
        fontSize=22, textColor=WHITE, fontName="Helvetica-Bold",
        spaceAfter=4, leading=26)
    sub_style = ParagraphStyle("TR_Sub", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#a78bfa"),
        fontName="Helvetica", spaceAfter=2)
    h2_style = ParagraphStyle("TR_H2", parent=styles["Heading2"],
        fontSize=13, textColor=PURPLE, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("TR_Body", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#1e293b"),
        fontName="Helvetica", leading=16)
    label_style = ParagraphStyle("TR_Label", parent=styles["Normal"],
        fontSize=9, textColor=MUTED, fontName="Helvetica")
    value_style = ParagraphStyle("TR_Value", parent=styles["Normal"],
        fontSize=11, textColor=DARK, fontName="Helvetica-Bold")

    # ── HEADER BLOCK ──
    rec   = get_recommendation(candidate.get("overall_score", 0))
    name  = candidate.get("candidate_name", "Candidate")
    score = candidate.get("overall_score", 0)

    header_data = [[
        Paragraph(f"TalentRank AI — Candidate Report", title_style),
        Paragraph(f"Rank #{rank} of {total}", sub_style),
    ]]
    header_table = Table(header_data, colWidths=[130*mm, 40*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK),
        ("ROWPADDING", (0,0), (-1,-1), 14),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",      (1,0), (1,0),   "RIGHT"),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ── CANDIDATE META ──
    meta_data = [
        [Paragraph("Candidate Name", label_style), Paragraph(name, value_style),
         Paragraph("Date Generated", label_style), Paragraph(datetime.now().strftime("%d %b %Y, %I:%M %p"), value_style)],
        [Paragraph("Final AI Score", label_style), Paragraph(f"{score}%", ParagraphStyle("S", parent=value_style, fontSize=16, textColor=PURPLE)),
         Paragraph("Recommendation", label_style), Paragraph(rec["label"], ParagraphStyle("R", parent=value_style, textColor=GREEN if score>=80 else (ORANGE if score>=65 else RED)))],
    ]
    meta_table = Table(meta_data, colWidths=[38*mm, 57*mm, 38*mm, 37*mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHT),
        ("ROWPADDING", (0,0), (-1,-1), 10),
        ("LINEBELOW",  (0,0), (-1,-2), 0.5, colors.HexColor("#e2e8f0")),
        ("ROUNDEDCORNERS", [8]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── SCORES TABLE ──
    story.append(Paragraph("Score Breakdown", h2_style))
    score_data = [
        ["Component", "Score", "Weight"],
        ["Semantic Similarity (Transformer AI)", f"{candidate.get('semantic_similarity',0)}%", "60%"],
        ["Skill Match",                           f"{candidate.get('skills_score',0)}%",       "25%"],
        ["Experience Match",                      f"{candidate.get('experience_score',0)}%",   "10%"],
        ["Education Match",                       f"{candidate.get('education_score',0)}%",    " 5%"],
        ["FINAL AI SCORE",                        f"{score}%",                                 "100%"],
    ]
    score_table = Table(score_data, colWidths=[95*mm, 35*mm, 40*mm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  PURPLE),
        ("TEXTCOLOR",   (0,0), (-1,0),  WHITE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  10),
        ("BACKGROUND",  (0,-1),(-1,-1), DARK),
        ("TEXTCOLOR",   (0,-1),(-1,-1), WHITE),
        ("FONTNAME",    (0,-1),(-1,-1), "Helvetica-Bold"),
        ("ROWPADDING",  (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [colors.white, LIGHT]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 10))

    # ── SKILL ANALYSIS ──
    story.append(Paragraph("Skill Gap Analysis", h2_style))
    matched  = candidate.get("matched_skills", [])
    missing  = candidate.get("missing_skills", [])
    additional = candidate.get("additional_skills", [])

    def skill_para(skills, color_hex):
        if not skills: return Paragraph("None identified", label_style)
        chips = "  ".join(skills[:15])
        return Paragraph(chips, ParagraphStyle("SC", parent=body_style,
                                               textColor=colors.HexColor(color_hex)))

    skill_data = [
        ["Matching Skills ✓",  "Missing Skills ✗",     "Additional Skills +"],
        [skill_para(matched, "#10b981"),
         skill_para(missing, "#f87171"),
         skill_para(additional, "#60a5fa")],
    ]
    skill_table = Table(skill_data, colWidths=[57*mm, 57*mm, 56*mm])
    skill_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 9),
        ("ROWPADDING", (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ]))
    story.append(skill_table)
    story.append(Spacer(1, 10))

    # ── AI ANALYSIS ──
    story.append(Paragraph("AI Analysis", h2_style))
    ai = candidate.get("ai_analysis", {})
    if ai.get("summary"):
        story.append(Paragraph(ai["summary"], body_style))
        story.append(Spacer(1, 6))

    if ai.get("strengths"):
        story.append(Paragraph("Strengths:", ParagraphStyle("SH", parent=body_style,
                                                             fontName="Helvetica-Bold")))
        for s in ai["strengths"]:
            story.append(Paragraph(f"  ✓  {s}", ParagraphStyle("SI", parent=body_style,
                                                                 textColor=GREEN, leftIndent=10)))
        story.append(Spacer(1, 4))

    if ai.get("improvements"):
        story.append(Paragraph("Areas for Improvement:", ParagraphStyle("IH", parent=body_style,
                                                                         fontName="Helvetica-Bold")))
        for imp in ai["improvements"]:
            story.append(Paragraph(f"  •  {imp}", ParagraphStyle("II", parent=body_style,
                                                                   textColor=ORANGE, leftIndent=10)))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated by TalentRank AI v3.0  •  {datetime.now().strftime('%d %b %Y %H:%M')}  •  Transformer-based Semantic Matching",
        ParagraphStyle("Footer", parent=label_style, alignment=1)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/rank", methods=["POST"])
def rank():
    body       = request.get_json(force=True, silent=True) or {}
    jd         = body.get("job_description", "").strip()
    candidates = body.get("candidates", [])

    if not jd:         return jsonify({"error": "job_description is required"}), 400
    if not candidates: return jsonify({"error": "candidates list is required"}), 400

    jd_skills = extract_skills(jd)
    results   = []

    for cand in candidates:
        name   = (cand.get("name")   or "Candidate").strip()
        resume = (cand.get("resume") or "").strip()
        if not resume: continue

        # ── AI PIPELINE ──
        sem_sim    = semantic_similarity(jd, resume)
        res_skills = extract_skills(resume)
        matched    = sorted(set(jd_skills) & set(res_skills))
        missing    = sorted(set(jd_skills) - set(res_skills))
        additional = get_additional_skills(jd_skills, res_skills)

        skill_sc   = compute_skill_score(jd_skills, res_skills)
        exp_sc     = compute_experience_score(jd, resume)
        edu_sc     = compute_education_score(resume)
        final_sc   = compute_final_score(sem_sim, skill_sc, exp_sc, edu_sc)
        rec        = get_recommendation(final_sc)
        yrs        = extract_years(resume)
        ai_analysis = gen_ai_analysis(name, final_sc, matched, missing,
                                       sem_sim, yrs, resume, jd)

        results.append({
            "candidate_name":      name,
            "overall_score":       final_sc,
            "semantic_similarity": round(sem_sim * 100, 1),
            "skills_score":        skill_sc,
            "experience_score":    exp_sc,
            "education_score":     edu_sc,
            "matched_skills":      [fmt_skill(s) for s in matched],
            "missing_skills":      [fmt_skill(s) for s in missing],
            "additional_skills":   [fmt_skill(s) for s in additional],
            "recommendation":      rec["label"],
            "recommendation_class":rec["class"],
            "recommendation_emoji":rec["emoji"],
            "ai_analysis":         ai_analysis,
            "years_experience":    yrs,
        })

    results.sort(key=lambda x: x["overall_score"], reverse=True)

    # Admin summary
    total     = len(results)
    rec_count = sum(1 for r in results if r["recommendation"] == "Recommended")
    con_count = sum(1 for r in results if r["recommendation"] == "Consider")
    not_count = sum(1 for r in results if r["recommendation"] == "Not Recommended")
    avg_score = round(sum(r["overall_score"] for r in results) / max(total, 1), 1)
    best      = results[0]["candidate_name"] if results else "N/A"

    return jsonify({
        "rankings": results,
        "jd_skills_extracted": [fmt_skill(s) for s in jd_skills],
        "model": "sentence-transformers/all-MiniLM-L6-v2 (Transformer)" if TRANSFORMER_SUPPORT else "TF-IDF Fallback",
        "transformer_active": TRANSFORMER_SUPPORT,
        "admin_summary": {
            "total":            total,
            "recommended":      rec_count,
            "consider":         con_count,
            "not_recommended":  not_count,
            "average_score":    avg_score,
            "best_candidate":   best,
        }
    })


@app.route("/api/parse-pdfs", methods=["POST"])
def parse_pdfs():
    if not PDF_SUPPORT:
        return jsonify({"error": "PyMuPDF not installed. Run: pip install PyMuPDF"}), 501
    files = request.files.getlist("resumes")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400
    results, errors = [], []
    for f in files:
        filename = f.filename or "unknown.pdf"
        raw_name = re.sub(r'\.pdf$', '', filename, flags=re.I)
        candidate_name = re.sub(r'[_\-]+', ' ', raw_name).strip().title()
        try:
            pdf_bytes = f.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = [page.get_text("text") for page in doc]
            doc.close()
            full_text = "\n".join(pages_text).strip()
            if not full_text:
                errors.append(f"{filename}: no readable text (may be scanned)")
                continue
            results.append({"filename": filename, "name": candidate_name,
                             "text": full_text, "pages": len(pages_text)})
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    return jsonify({"parsed": results, "errors": errors, "total": len(results)})


@app.route("/api/export/pdf", methods=["POST"])
def export_pdf():
    if not REPORTLAB_SUPPORT:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 501
    body = request.get_json(force=True, silent=True) or {}
    candidate = body.get("candidate", {})
    rank  = body.get("rank", 1)
    total = body.get("total", 1)
    pdf_bytes = generate_pdf_report(candidate, rank, total)
    if not pdf_bytes:
        return jsonify({"error": "PDF generation failed"}), 500
    name = re.sub(r'[^a-zA-Z0-9_\- ]', '', candidate.get("candidate_name", "report")).replace(" ", "_")
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                     as_attachment=True, download_name=f"TalentRank_{name}.pdf")


@app.route("/api/export/csv", methods=["POST"])
def export_csv():
    body    = request.get_json(force=True, silent=True) or {}
    results = body.get("rankings", [])
    buf     = io.StringIO()
    writer  = csv.writer(buf)
    writer.writerow(["Rank","Candidate","Final AI Score","Semantic Similarity",
                     "Skill Match","Experience","Education","Recommendation",
                     "Matched Skills","Missing Skills"])
    for i, r in enumerate(results, 1):
        writer.writerow([
            i, r.get("candidate_name"), f"{r.get('overall_score')}%",
            f"{r.get('semantic_similarity')}%", f"{r.get('skills_score')}%",
            f"{r.get('experience_score')}%",    f"{r.get('education_score')}%",
            r.get("recommendation"),
            "; ".join(r.get("matched_skills",[])),
            "; ".join(r.get("missing_skills",[])),
        ])
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype="text/csv", as_attachment=True,
                     download_name="TalentRank_Results.csv")


@app.route("/api/export/excel", methods=["POST"])
def export_excel():
    if not PANDAS_SUPPORT:
        return jsonify({"error": "pandas/openpyxl not installed"}), 501
    body    = request.get_json(force=True, silent=True) or {}
    results = body.get("rankings", [])
    rows = []
    for i, r in enumerate(results, 1):
        rows.append({
            "Rank":                i,
            "Candidate":           r.get("candidate_name"),
            "Final AI Score (%)":  r.get("overall_score"),
            "Semantic Sim (%)":    r.get("semantic_similarity"),
            "Skill Match (%)":     r.get("skills_score"),
            "Experience (%)":      r.get("experience_score"),
            "Education (%)":       r.get("education_score"),
            "Recommendation":      r.get("recommendation"),
            "Matched Skills":      "; ".join(r.get("matched_skills", [])),
            "Missing Skills":      "; ".join(r.get("missing_skills", [])),
        })
    df  = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Candidates")
    buf.seek(0)
    return send_file(buf, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="TalentRank_Results.xlsx")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  "sentence-transformers/all-MiniLM-L6-v2" if TRANSFORMER_SUPPORT else "TF-IDF Fallback",
        "transformer_active": TRANSFORMER_SUPPORT,
        "pdf_support":  PDF_SUPPORT,
        "reportlab":    REPORTLAB_SUPPORT,
        "pandas":       PANDAS_SUPPORT,
    })


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║          🧠  TalentRank AI  v3.0  (Upgraded)        ║")
    print("  ╠══════════════════════════════════════════════════════╣")
    print(f"  ║  AI Engine : {'Transformer (all-MiniLM-L6-v2)' if TRANSFORMER_SUPPORT else 'TF-IDF (Fallback)      ':<38}║")
    print(f"  ║  PDF Parse : {'PyMuPDF ✅' if PDF_SUPPORT else 'Not Installed ❌':<38}║")
    print(f"  ║  PDF Report: {'ReportLab ✅' if REPORTLAB_SUPPORT else 'Not Installed ❌':<38}║")
    print(f"  ║  Excel Exp : {'Pandas ✅' if PANDAS_SUPPORT else 'Not Installed ❌':<38}║")
    print("  ║  ✅ Run ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print("  → Open http://localhost:5000")
    print()
    app.run(debug=False, port=5000, host="0.0.0.0")
