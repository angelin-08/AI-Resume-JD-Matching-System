
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_FILE = OUTPUT_DIR / "extracted_documents.csv"
MANIFEST_FILE = BASE_DIR / "ground_truth_pairs.csv"

# Initial weights; we'll evaluate and refine these.
TFIDF_WEIGHT = 0.20
SEMANTIC_WEIGHT = 0.50
SKILL_WEIGHT = 0.30

# Skills used for this initial prototype.
SKILLS = [
    "python", "sql", "pandas", "numpy", "excel", "power bi",
    "scikit-learn", "machine learning", "classification",
    "feature engineering", "fastapi", "flask", "rest api",
    "postgresql", "git", "pytest", "javascript", "react",
    "html", "css", "node.js", "express", "selenium",
    "api testing", "requirements gathering", "user stories",
    "process mapping", "uat", "linux", "network fundamentals",
    "siem", "incident response", "vulnerability assessment",
    "nlp", "sentence embeddings", "vector search", "rag",
    "llm api", "prompt engineering", "aws", "kafka",
    "redis", "docker", "system design", "statistics",
    "data cleaning", "data visualization", "exploratory data analysis",
    "cross-validation", "model evaluation", "model deployment",
    "stakeholder communication", "acceptance criteria",
    "regression testing", "bug reporting", "access control",
]


def find_skills(text):
    """Return skills found in the supplied text."""
    text = str(text).lower()
    found = []

    for skill in SKILLS:
        if skill in text:
            found.append(skill)

    return sorted(set(found))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run extract_text.py first to create extracted_documents.csv."
        )

    documents = pd.read_csv(INPUT_FILE).fillna("")

    resumes = documents[
        documents["document_type"] == "RESUME"
    ].copy()

    jds = documents[
        documents["document_type"] == "JD"
    ].copy()

    if resumes.empty or jds.empty:
        raise ValueError("Resume or JD documents are missing.")

    print(f"Resumes loaded: {len(resumes)}")
    print(f"Job descriptions loaded: {len(jds)}")
    print(f"Pairs to compare: {len(resumes) * len(jds)}")

    # Combine both groups so lexical similarities share one vocabulary.
    all_texts = (
        resumes["extracted_text"].tolist()
        + jds["extracted_text"].tolist()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)

    resume_tfidf = tfidf_matrix[:len(resumes)]
    jd_tfidf = tfidf_matrix[len(resumes):]

    lexical_matrix = cosine_similarity(
        resume_tfidf, jd_tfidf
    )

    print("\nLoading semantic model...")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    resume_texts = resumes["extracted_text"].tolist()
    jd_texts = jds["extracted_text"].tolist()

    resume_embeddings = model.encode(
        resume_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    jd_embeddings = model.encode(
        jd_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    semantic_matrix = util.cos_sim(
        resume_embeddings, jd_embeddings
    ).cpu().numpy()

    resume_skills = [
        find_skills(text) for text in resume_texts
    ]
    jd_skills = [
        find_skills(text) for text in jd_texts
    ]

    results = []

    for i, (_, resume) in enumerate(resumes.iterrows()):
        for j, (_, jd) in enumerate(jds.iterrows()):
            resume_skill_set = set(resume_skills[i])
            jd_skill_set = set(jd_skills[j])

            matched = sorted(resume_skill_set & jd_skill_set)
            missing = sorted(jd_skill_set - resume_skill_set)

            # Required-skill coverage proxy for this prototype.
            skill_score = (
                len(matched) / len(jd_skill_set)
                if jd_skill_set else 0.0
            )

            lexical_score = float(lexical_matrix[i, j])

            # Convert cosine similarity to a 0-1 range.
            semantic_score = max(
                0.0, min(1.0, float(semantic_matrix[i, j]))
            )

            final_score = 100 * (
                TFIDF_WEIGHT * lexical_score
                + SEMANTIC_WEIGHT * semantic_score
                + SKILL_WEIGHT * skill_score
            )

            results.append({
                "resume_id": resume["document_id"],
                "resume_filename": resume["filename"],
                "jd_id": jd["document_id"],
                "jd_filename": jd["filename"],
                "match_score": round(final_score, 2),
                "lexical_similarity": round(lexical_score * 100, 2),
                "semantic_similarity": round(semantic_score * 100, 2),
                "skill_coverage": round(skill_score * 100, 2),
                "matched_skills": ", ".join(matched),
                "missing_skills": ", ".join(missing),
                "strong_match": final_score >= 60,
            })

    details = pd.DataFrame(results)
    details = details.sort_values(
        ["resume_id", "match_score"],
        ascending=[True, False],
    )

    # Table 1: Resume -> matching JDs.
    resume_rows = []

    for resume_id, group in details.groupby("resume_id", sort=True):
        matches = group[group["strong_match"]].sort_values(
            "match_score", ascending=False
        )

        resume_rows.append({
            "resume_id": resume_id,
            "matching_jds": ", ".join(matches["jd_id"]),
            "match_scores": "; ".join(
                f"{row.jd_id}: {row.match_score:.2f}%"
                for row in matches.itertuples()
            ),
            "best_match_jd": (
                group.iloc[0]["jd_id"] if not group.empty else ""
            ),
            "best_match_score": (
                group.iloc[0]["match_score"] if not group.empty else 0
            ),
        })

    resume_report = pd.DataFrame(resume_rows)

   
    # Table 2: JD -> matching resumes.
    jd_rows = []

    for jd_id, group in details.groupby("jd_id", sort=True):
        # Sort every JD's comparisons from highest to lowest score.
        ranked = group.sort_values(
            "match_score", ascending=False
        ).reset_index(drop=True)

        # Only include matches meeting the current threshold.
        matches = ranked[ranked["strong_match"]]

        # Best match must come from the highest-scoring comparison.
        best = ranked.iloc[0]

        jd_rows.append({
            "jd_id": jd_id,
            "matching_resumes": ", ".join(matches["resume_id"]),
            "match_scores": "; ".join(
                f"{row.resume_id}: {row.match_score:.2f}%"
                for row in matches.itertuples()
            ),
            "best_match_resume": best["resume_id"],
            "best_match_score": best["match_score"],
        })

    jd_report = pd.DataFrame(jd_rows)


    # Export all required tables to one workbook.
    report_path = OUTPUT_DIR / "resume_jd_matching_report.xlsx"

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        resume_report.to_excel(
            writer, sheet_name="Resume-wise Matches", index=False
        )
        jd_report.to_excel(
            writer, sheet_name="JD-wise Matches", index=False
        )
        details.to_excel(
            writer, sheet_name="All Comparisons", index=False
        )

    print("\nMATCHING COMPLETE")
    print(f"Total comparisons: {len(details)}")
    print(f"Report saved to: {report_path}")
    print("\nTop 10 comparisons:")
    print(
        details.nlargest(10, "match_score")[
            ["resume_id", "jd_id", "match_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
