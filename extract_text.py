from pathlib import Path

import pandas as pd
import pymupdf

# Locate the project folders
BASE_DIR = Path(__file__).resolve().parent
RESUMES_DIR = BASE_DIR / "resumes"
JDS_DIR = BASE_DIR / "job_descriptions"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def extract_pdf_text(pdf_path):
    """Extract text from all pages of a PDF."""
    with pymupdf.open(pdf_path) as document:
        text = "\n".join(page.get_text() for page in document).strip()
        page_count = len(document)

    return text, page_count


def process_folder(folder, document_type):
    """Extract text from every PDF in a folder."""
    records = []
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"WARNING: No PDFs found in {folder}")
        return records

    for pdf_path in pdf_files:
        try:
            text, page_count = extract_pdf_text(pdf_path)

            status = "SUCCESS" if text else "EMPTY TEXT"

            records.append({
                "document_type": document_type,
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "page_count": page_count,
                "character_count": len(text),
                "extraction_status": status,
                "extracted_text": text,
            })

            print(
                f"[{status}] {pdf_path.name}: "
                f"{len(text)} characters"
            )

        except Exception as error:
            records.append({
                "document_type": document_type,
                "document_id": pdf_path.stem,
                "filename": pdf_path.name,
                "page_count": 0,
                "character_count": 0,
                "extraction_status": f"ERROR: {error}",
                "extracted_text": "",
            })

            print(f"[ERROR] {pdf_path.name}: {error}")

    return records


def main():
    print("\nRESUME–JD DOCUMENT EXTRACTION")
    print("=" * 45)

    records = []
    records.extend(process_folder(RESUMES_DIR, "RESUME"))
    records.extend(process_folder(JDS_DIR, "JD"))

    if not records:
        print("No PDFs were processed. Check the folder paths.")
        return

    results = pd.DataFrame(records)

    # Save extracted text for later matching.
    results.to_csv(
        OUTPUT_DIR / "extracted_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save a concise quality-check report.
    results.drop(columns=["extracted_text"]).to_csv(
        OUTPUT_DIR / "extraction_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 45)
    print("EXTRACTION SUMMARY")
    print("=" * 45)
    print(f"Total PDFs processed: {len(results)}")
    print(f"Resumes: {(results['document_type'] == 'RESUME').sum()}")
    print(f"Job descriptions: {(results['document_type'] == 'JD').sum()}")
    print(f"Successful: {(results['extraction_status'] == 'SUCCESS').sum()}")
    print(f"Empty text: {(results['extraction_status'] == 'EMPTY TEXT').sum()}")
    print(f"Errors: {results['extraction_status'].str.startswith('ERROR').sum()}")
    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()