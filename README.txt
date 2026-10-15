FRESH SYNTHETIC RESUME–JD MATCHING DATASET
============================================

Contents
- resumes/: 24 fictional resume PDFs (3 profiles for each of 8 role families)
- job_descriptions/: 8 fictional JD PDFs (one per role family)
- dataset_manifest.csv: resume IDs, file names and designed role labels
- jd_manifest.csv: JD IDs, file names and designed role labels
- ground_truth_pairs.csv: all 192 resume–JD pairs with an intended relevance label

Role families
1. Data Analyst
2. Python Backend Developer
3. Machine Learning Engineer
4. NLP / Generative AI Engineer
5. Full Stack Developer
6. QA Automation Engineer
7. Business Analyst
8. Cybersecurity Analyst

Important evaluation rules
- All names, organizations, emails and profiles are fictional. Emails use example.test.
- Expected positive matches were deliberately designed by role family. This makes the dataset controlled and reproducible, but it is not a substitute for human-labelled real-world data.
- Keep ground_truth_pairs.csv and the role-label columns out of the matching model's input. Use them only after scoring, to evaluate results.
- A match score is not a hiring probability. Set a match threshold using validation results and inspect false positives and false negatives.
- Because the dataset has 3 intended positive resumes per JD, report precision, recall, F1 and per-JD top-k results; accuracy alone can be misleading.
- Before using this in a real hiring workflow, have people review labels and test for bias, privacy, and job-related validity.
