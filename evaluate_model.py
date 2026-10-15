from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_FILE = OUTPUT_DIR / "resume_jd_matching_report.xlsx"

GROUND_TRUTH_FILE = BASE_DIR / "ground_truth_pairs.csv"
RESUME_MANIFEST = BASE_DIR / "dataset_manifest.csv"
JD_MANIFEST = BASE_DIR / "jd_manifest.csv"

THRESHOLD = 60.0


def main():
    print("Loading evaluation data...")

    # Load the generated scores and the expected relevance labels.
    scores = pd.read_excel(
        REPORT_FILE, sheet_name="All Comparisons"
    )
    truth = pd.read_csv(GROUND_TRUTH_FILE)
    resume_manifest = pd.read_csv(RESUME_MANIFEST)
    jd_manifest = pd.read_csv(JD_MANIFEST)

    # Translate R01 -> Resume_01, etc.
    resume_map = {
        row["resume_id"]: (
            f"Resume_{int(str(row['filename']).split('_')[1].split('.')[0]):02d}"
        )
        for _, row in resume_manifest.iterrows()
    }

    # Translate JD01 -> JD_01_DA, etc.
    jd_map = {
        row["jd_id"]: (
            f"JD_{str(row['jd_id'])[2:]}_{row['designed_role_code']}"
        )
        for _, row in jd_manifest.iterrows()
    }

    truth["resume_report_id"] = truth["resume_id"].map(resume_map)
    truth["jd_report_id"] = truth["jd_id"].map(jd_map)

    if truth[["resume_report_id", "jd_report_id"]].isna().any().any():
        raise ValueError(
            "Some ground-truth IDs could not be mapped. "
            "Check the manifest column names and IDs."
        )

    # Join each score with its expected label.
    merged = scores.merge(
        truth,
        left_on=["resume_id", "jd_id"],
        right_on=["resume_report_id", "jd_report_id"],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(truth):
        raise ValueError(
            f"Only {len(merged)} of {len(truth)} labelled pairs matched."
        )

    
    y_true = merged["relevant_label"].astype(int).tolist()
    y_pred = (
        merged["match_score"] >= THRESHOLD
    ).astype(int).tolist()

    # Calculate metrics using plain Python.
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    print("\nEVALUATION RESULTS")
    print("=" * 45)
    print(f"Labelled pairs evaluated: {total}")
    print(f"Score threshold: {THRESHOLD:.1f}%")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")

    print("\nCONFUSION MATRIX")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")


    # Check whether the highest-scoring JD has the designed role.
    resume_roles = resume_manifest.set_index(
        "resume_id"
    )["designed_role_code"].to_dict()

    jd_roles = jd_manifest.set_index(
        "jd_id"
    )["designed_role_code"].to_dict()


    role_correct = []

    for resume_id, group in scores.groupby("resume_id"):
        best = group.loc[group["match_score"].idxmax()]

        resume_number = int(resume_id.split("_")[1])
        truth_resume_id = f"R{resume_number:02d}"

        best_jd_number = int(best["jd_id"].split("_")[1])
        truth_jd_id = f"JD{best_jd_number:02d}"

        expected_role = resume_roles.get(truth_resume_id)
        predicted_role = jd_roles.get(truth_jd_id)

        role_correct.append(expected_role == predicted_role)

    role_accuracy = sum(role_correct) / len(role_correct)

    print(
        f"\nDesigned-role top-match accuracy: "
        f"{role_accuracy:.3f} ({sum(role_correct)}/{len(role_correct)})"
    )

    # Save pair-level predictions for inspection.
    merged["predicted_relevant"] = y_pred
    merged["prediction_correct"] = (
        merged["predicted_relevant"] == merged["relevant_label"]
    )

    result_path = OUTPUT_DIR / "model_evaluation_results.xlsx"

    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        merged.to_excel(
            writer, sheet_name="Pair Evaluation", index=False
        )

        pd.DataFrame({
            "metric": [
                "Threshold",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 score",
                "Designed-role top-match accuracy",
                "True positives",
                "False positives",
                "True negatives",
                "False negatives",
            ],
            "value": [
                THRESHOLD,
                accuracy,
                precision,
                recall,
                f1,
                role_accuracy,
                tp, fp, tn, fn,
            ],
        }).to_excel(writer, sheet_name="Summary", index=False)

    print(f"\nDetailed evaluation saved to: {result_path}")


if __name__ == "__main__":
    main()
