# 🧠 TalentRank AI v3.0 — AI-Powered Resume Screening System

> **Upgraded from TF-IDF → Transformer Embeddings (all-MiniLM-L6-v2)**
> Fully offline · No API keys · Suitable for AI/ML college project viva

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py

# 3. Open browser
http://localhost:5000
```

The first run will auto-download the `all-MiniLM-L6-v2` model (~90MB).
After that it works 100% offline.

---

## 🤖 AI Engine

| Component | Technology |
|-----------|-----------|
| **Semantic Matching** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Embeddings** | 384-dim dense vectors |
| **Similarity** | Cosine similarity (sklearn) |
| **Fallback** | TF-IDF (if transformers unavailable) |

---

## 📊 Hybrid Scoring Formula

```
Final AI Score = 60% Semantic Similarity
               + 25% Skill Match
               + 10% Experience Match
               +  5% Education Match
```

---

## ✨ Features

- ✅ Upload 1–20 resume PDFs simultaneously  
- ✅ Automatic candidate ranking (descending)  
- ✅ Skill Gap Analysis (Matching / Missing / Additional)  
- ✅ AI-generated analysis panel with Strengths & Improvements  
- ✅ Recommendation badges: Recommended / Consider / Not Recommended  
- ✅ Interactive dashboard with score cards  
- ✅ Bar chart + Doughnut chart visualizations  
- ✅ Top 3 podium with medal icons  
- ✅ Full ranking table with progress bars  
- ✅ PDF report generation (ReportLab)  
- ✅ CSV & Excel export  
- ✅ Recruiter Admin Summary panel  
- ✅ Glassmorphism UI with hover animations  
- ✅ Mobile responsive layout  

---

## 📁 Project Structure

```
talentrank/
├── app.py              ← Flask backend + AI engine
├── requirements.txt    ← All dependencies
├── README.md           ← This file
└── static/
    └── index.html      ← Complete frontend (single file)
```

---

## 🎓 Viva Q&A Cheat Sheet

**Q: What AI model are you using?**  
A: `sentence-transformers/all-MiniLM-L6-v2` — a pre-trained Transformer model from Hugging Face that generates 384-dimensional semantic embeddings.

**Q: How is semantic similarity computed?**  
A: Both resume and JD texts are encoded into embedding vectors. Cosine similarity between these vectors gives a score from 0 to 1 (then ×100 for %).

**Q: Why Transformers over TF-IDF?**  
A: TF-IDF is keyword-based. Transformers understand meaning — "ML engineer" and "machine learning developer" would score low in TF-IDF but high semantically.

**Q: How is the Final Score calculated?**  
A: Weighted average: 60% Semantic + 25% Skill keyword overlap + 10% Experience match + 5% Education level.

**Q: How are skills extracted?**  
A: Regex-based matching against a dictionary of 120+ tech skills (case-insensitive, whole-word matching).

**Q: Is this truly offline?**  
A: Yes. After the one-time model download (~90MB), no internet is needed.
