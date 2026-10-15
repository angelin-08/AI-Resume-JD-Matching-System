"""
AI Resume–Job Description Matching Dashboard
Main Streamlit Application Entry Point
"""

from pathlib import Path
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.data_engine import load_all_data, compute_dynamic_views
from utils.ats_engine import (
    extract_text_from_pdf_bytes,
    analyze_sections,
    compute_live_match,
    compute_ats_readiness_assessment,
)
from utils.excel_generator import generate_excel_report
from utils.ui_components import (
    get_custom_css,
    render_kpi,
    render_skill_chips_html,
    get_plotly_layout,
)

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & CACHING
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume–JD Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data(show_spinner=False)
def get_cached_project_data():
    """Load and cache manifests, texts, and baseline reports."""
    return load_all_data(BASE_DIR)


# Load data with error handling
try:
    project_data = get_cached_project_data()
except Exception as e:
    st.error(f"Failed to load project dataset files: {str(e)}")
    st.info(
        "Ensure that dataset_manifest.csv, jd_manifest.csv, and outputs/ directory "
        "exist in the project folder."
    )
    st.stop()

# -------------------------------------------------------------
# 2. SIDEBAR NAVIGATION & THEME CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 Talent Intelligence")
    st.markdown(
        "<span style='font-size:12px; color:gray;'>AI Resume–JD Matching & ATS Diagnostics</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Dark / Light Theme Toggle
    is_dark = st.toggle("🌙 Dark Mode", value=False)

    # Inject Custom CSS
    st.markdown(get_custom_css(is_dark=is_dark), unsafe_allow_html=True)

    # Navigation Radio
    page_selection = st.radio(
        "Navigation",
        [
            "📊 Overview Dashboard",
            "📄 Resume-Wise Matching",
            "💼 JD-Wise Matching",
            "🔍 Detailed Matching Analysis",
            "⚡ Resume Checker / ATS Readiness",
            "📥 Reports and Downloads",
        ],
        index=1,  # Default to Resume-Wise Matching as primary deliverable
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Matching Configuration")

    # Configurable Threshold Slider (Default 60.0%)
    match_threshold = st.slider(
        "Match Qualifying Threshold (%)",
        min_value=10.0,
        max_value=90.0,
        value=60.0,
        step=1.0,
        help="Pairs with match scores at or above this threshold qualify as strong matches.",
    )

    # Calculate dynamic views based on threshold
    dynamic_views = compute_dynamic_views(project_data, threshold=match_threshold)
    stats = dynamic_views["summary_stats"]

    st.markdown("---")
    st.markdown("#### 📈 Dataset Quick Status")
    st.markdown(
        f"""
        - **Resumes:** {stats['total_resumes']}
        - **Job Descriptions:** {stats['total_jds']}
        - **Comparisons:** {stats['total_comparisons']}
        - **Qualifying Matches:** `{stats['qualifying_matches']}`
        - **Zero-Match Resumes:** `{stats['resumes_without_matches']}`
        """
    )

    st.caption("Fresh Synthetic Resume–JD Matching Dataset v1.0")

# -------------------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# -------------------------------------------------------------
if page_selection == "📊 Overview Dashboard":
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Executive Overview Dashboard</div>
            <div class="page-desc">Comprehensive analytics on candidate-role compatibility, score distributions, and cohort alignments across the 24 fictional resumes and 8 job descriptions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: KPI Metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            render_kpi("Total Resumes", str(stats["total_resumes"]), "8 Role Families", "#4F46E5"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_kpi("Job Descriptions", str(stats["total_jds"]), "1 per Role Family", "#7C3AED"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_kpi("Total Pairs", str(stats["total_comparisons"]), "24 × 8 Matrix", "#2563EB"),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            render_kpi(
                "Qualifying Matches",
                str(stats["qualifying_matches"]),
                f"Score ≥ {match_threshold:.0f}%",
                "#10B981",
            ),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            render_kpi(
                "Zero-Match Resumes",
                str(stats["resumes_without_matches"]),
                f"Below {match_threshold:.0f}%",
                "#F59E0B" if stats["resumes_without_matches"] > 0 else "#10B981",
            ),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            render_kpi(
                "Avg Match Score",
                f"{stats['avg_match_score']:.1f}%",
                f"Qualifying: {stats['avg_qualifying_score']:.1f}%",
                "#0D9488",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts
    col_chart_left, col_chart_right = st.columns(2)

    with col_chart_left:
        # Chart 1: Score Distribution Histogram with threshold line
        comp_df = dynamic_views["comparisons_df"]
        fig_hist = px.histogram(
            comp_df,
            x="match_score",
            nbins=25,
            color="is_qualifying",
            color_discrete_map={True: "#10B981", False: "#6366F1"},
            labels={"match_score": "Match Score (%)", "is_qualifying": "Qualifies"},
            title="Match Score Distribution Across All 192 Comparisons",
        )
        fig_hist.add_vline(
            x=match_threshold,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Threshold ({match_threshold:.0f}%)",
            annotation_position="top right",
        )
        fig_hist.update_layout(get_plotly_layout("Match Score Distribution", is_dark))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart_right:
        # Chart 2: Number of Matches per Resume
        resume_df = dynamic_views["resume_wise_df"]
        fig_res = px.bar(
            resume_df,
            x="resume_code",
            y="qualifying_count",
            color="qualifying_count",
            color_continuous_scale="Purples",
            labels={"resume_code": "Resume ID", "qualifying_count": "Matches Count"},
            title=f"Qualifying Matches per Resume (Threshold: {match_threshold:.0f}%)",
        )
        fig_res.update_layout(get_plotly_layout("Qualifying Matches per Resume", is_dark, showlegend=False))
        st.plotly_chart(fig_res, use_container_width=True)

    col_chart_b1, col_chart_b2 = st.columns(2)

    with col_chart_b1:
        # Chart 3: Candidate Supply per Job Description
        jd_df = dynamic_views["jd_wise_df"]
        fig_jd = px.bar(
            jd_df,
            x="role_title",
            y="qualifying_count",
            color="role_title",
            labels={"role_title": "Role Title", "qualifying_count": "Qualified Candidates"},
            title="Candidate Supply by Job Description",
        )
        fig_jd.update_layout(
            get_plotly_layout("Candidate Supply by Job Description", is_dark, showlegend=False)
        )
        st.plotly_chart(fig_jd, use_container_width=True)

    with col_chart_b2:
        # Chart 4: Role-Wise Matching Affinity Heatmap
        matrix_df = (
            comp_df.groupby(["resume_role", "jd_role"])["match_score"]
            .mean()
            .reset_index()
            .pivot(index="resume_role", columns="jd_role", values="match_score")
        )
        fig_heat = px.imshow(
            matrix_df,
            labels=dict(x="Job Description Role", y="Resume Candidate Role", color="Avg Score (%)"),
            x=matrix_df.columns,
            y=matrix_df.index,
            color_continuous_scale="Blues",
            title="Role Family Compatibility Matrix (Mean Score %)",
            text_auto=".1f",
        )
        fig_heat.update_layout(
            get_plotly_layout("Role Compatibility Matrix", is_dark, showlegend=False, height=400)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # Leaderboard & Benchmark Section
    st.markdown("#### 🏆 Top 10 Scoring Matches")
    top_10 = comp_df.nlargest(10, "match_score")[
        [
            "resume_code",
            "resume_role",
            "jd_code",
            "jd_role",
            "match_score",
            "lexical_similarity",
            "semantic_similarity",
            "skill_coverage",
        ]
    ].rename(
        columns={
            "resume_code": "Resume ID",
            "resume_role": "Candidate Role",
            "jd_code": "JD ID",
            "jd_role": "Job Role",
            "match_score": "Final Score (%)",
            "lexical_similarity": "TF-IDF (%)",
            "semantic_similarity": "Semantic (%)",
            "skill_coverage": "Skill Match (%)",
        }
    )
    st.dataframe(top_10.style.format({
        "Final Score (%)": "{:.2f}%",
        "TF-IDF (%)": "{:.2f}%",
        "Semantic (%)": "{:.2f}%",
        "Skill Match (%)": "{:.2f}%",
    }), use_container_width=True)

    # Benchmark Quality and Limitations Callout
    eval_df = project_data.get("eval_summary_df")
    with st.expander("🔬 Model Evaluation Benchmark Results & Verification"):
        st.markdown(
            """
            The prototype matching model was evaluated against `ground_truth_pairs.csv`
            where 3 intended positive resumes were designed for each of the 8 role families (24 intended positives total).
            """
        )
        if eval_df is not None:
            col_e1, col_e2 = st.columns([1, 2])
            with col_e1:
                st.dataframe(eval_df, hide_index=True, use_container_width=True)
            with col_e2:
                st.markdown(
                    """
                    **Key Evaluation Findings:**
                    - **Accuracy:** 98.44% across all 192 pairwise decisions.
                    - **Precision:** 100.00% (No false positive cross-role matches at 60.0% threshold).
                    - **Recall:** 87.50% (21 of 24 designed positives detected at 60.0% threshold).
                    - **Designed-Role Top-Match Accuracy:** 100.00% (Every resume's #1 ranked JD matches its designed role family).
                    """
                )
        st.markdown(
            """
            <div class="custom-callout">
                <strong>Evaluation Disclaimer:</strong> These metrics reflect performance on a controlled synthetic dataset designed by role family. They serve as reproducible architectural verification and must <em>not</em> be construed as proof of real-world recruitment efficacy. Actual hiring requires human domain expertise and rigorous bias auditing.
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# PAGE 2: RESUME-WISE MATCHING (PRIMARY DELIVERABLE)
# -------------------------------------------------------------
elif page_selection == "📄 Resume-Wise Matching":
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Resume-Wise Matching Report</div>
            <div class="page-desc">Primary Deliverable: For every resume in the dataset, inspect all qualifying job descriptions meeting the threshold. Every resume is included, even when no qualifying matches exist.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resume_wise_df = dynamic_views["resume_wise_df"]
    comp_df = dynamic_views["comparisons_df"]

    # Filter Bar
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 2])

    with f_col1:
        role_filter = st.selectbox(
            "Filter Candidate Role",
            ["All Roles"] + project_data["all_roles"],
            index=0,
        )
    with f_col2:
        status_filter = st.selectbox(
            "Filter Qualification Status",
            ["All Resumes (24)", "Qualifying Resumes Only", "Zero-Match Resumes Only"],
            index=0,
        )
    with f_col3:
        search_query = st.text_input("🔍 Search Resume ID or Filename", "").strip().lower()
    with f_col4:
        sort_by = st.selectbox(
            "Sort Order",
            [
                "Best Match Score (Desc)",
                "Qualifying Count (Desc)",
                "Resume ID (Asc)",
                "Resume ID (Desc)",
            ],
            index=0,
        )

    # Apply Filters
    filtered_resumes = resume_wise_df.copy()
    if role_filter != "All Roles":
        filtered_resumes = filtered_resumes[filtered_resumes["candidate_role"] == role_filter]

    if status_filter == "Qualifying Resumes Only":
        filtered_resumes = filtered_resumes[filtered_resumes["qualifying_count"] > 0]
    elif status_filter == "Zero-Match Resumes Only":
        filtered_resumes = filtered_resumes[filtered_resumes["qualifying_count"] == 0]

    if search_query:
        filtered_resumes = filtered_resumes[
            filtered_resumes["resume_code"].str.lower().str.contains(search_query)
            | filtered_resumes["resume_filename"].str.lower().str.contains(search_query)
            | filtered_resumes["candidate_role"].str.lower().str.contains(search_query)
        ]

    # Sorting
    if sort_by == "Best Match Score (Desc)":
        filtered_resumes = filtered_resumes.sort_values("best_match_score", ascending=False)
    elif sort_by == "Qualifying Count (Desc)":
        filtered_resumes = filtered_resumes.sort_values(
            ["qualifying_count", "best_match_score"], ascending=[False, False]
        )
    elif sort_by == "Resume ID (Asc)":
        filtered_resumes = filtered_resumes.sort_values("resume_code", ascending=True)
    elif sort_by == "Resume ID (Desc)":
        filtered_resumes = filtered_resumes.sort_values("resume_code", ascending=False)

    # Status Banner
    qual_count = int((filtered_resumes["qualifying_count"] > 0).sum())
    total_shown = len(filtered_resumes)
    unqual_count = total_shown - qual_count

    st.markdown(
        f"""
        <div class="custom-callout">
            Showing <strong>{total_shown}</strong> of 24 resumes &nbsp;|&nbsp;
            <strong>{qual_count}</strong> qualified &nbsp;|&nbsp;
            <strong>{unqual_count}</strong> with zero qualifying matches at <strong>{match_threshold:.1f}% threshold</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Master Table Display
    display_table = filtered_resumes[
        [
            "resume_code",
            "resume_filename",
            "candidate_role",
            "matching_jds",
            "match_scores",
            "qualifying_count",
            "best_match_jd",
            "best_match_score",
            "status",
        ]
    ].rename(
        columns={
            "resume_code": "Resume ID",
            "resume_filename": "Filename",
            "candidate_role": "Designed Role",
            "matching_jds": "Matching Job Descriptions",
            "match_scores": "Matching Scores",
            "qualifying_count": "Qualifying Count",
            "best_match_jd": "Best Match JD",
            "best_match_score": "Top Score (%)",
            "status": "Match Status",
        }
    )

    st.dataframe(
        display_table.style.format({"Top Score (%)": "{:.2f}%"}),
        use_container_width=True,
        height=380,
    )

    # Download Buttons for Resume-Wise Report
    col_d1, col_d2, _ = st.columns([2, 2, 4])
    with col_d1:
        # Full dynamic excel report
        excel_data = generate_excel_report(project_data, dynamic_views)
        st.download_button(
            label="📥 Download Full Excel Report (.xlsx)",
            data=excel_data,
            file_name=f"resume_jd_matching_report_threshold_{int(match_threshold)}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col_d2:
        csv_buffer = filtered_resumes.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Download Filtered Table (.csv)",
            data=csv_buffer,
            file_name=f"resume_wise_matches_filtered_{int(match_threshold)}.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # Resume Drill-Down Section: Inspect Against All 8 JDs
    # -------------------------------------------------------------
    st.markdown("### 🔎 Detailed Resume Drill-Down")
    st.markdown(
        "Select any resume to inspect its complete ranking across all 8 job descriptions, score components, and matched/missing skill gaps."
    )

    resume_options = [
        f"{r['resume_code']} - {r['resume_filename']} ({r['candidate_role']})"
        for _, r in resume_wise_df.iterrows()
    ]
    selected_resume_str = st.selectbox("Select Resume Profile", resume_options, index=0)
    selected_r_code = selected_resume_str.split(" - ")[0].strip()

    # Get data for selected resume
    r_manifest = project_data["resume_manifest"]
    r_info = r_manifest[r_manifest["resume_id"] == selected_r_code].iloc[0]
    r_stem = Path(r_info["filename"]).stem
    r_doc_meta = project_data["doc_text_map"].get(r_stem, {})
    r_text = r_doc_meta.get("text", "")

    # Profile Summary
    col_p1, col_p2 = st.columns([1, 1])

    with col_p1:
        st.markdown(
            f"""
            <div class="analytics-card">
                <h4 style="margin-top:0;">👤 Candidate Profile: {selected_r_code}</h4>
                <p><strong>Filename:</strong> <code>{r_info['filename']}</code></p>
                <p><strong>Designed Role:</strong> {r_info['designed_role']}</p>
                <p><strong>Document Size:</strong> {r_doc_meta.get('character_count', 0)} chars ({r_doc_meta.get('page_count', 1)} page)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        r_comps_all = comp_df[comp_df["resume_code"] == selected_r_code].sort_values(
            "match_score", ascending=False
        )
        best_comp = r_comps_all.iloc[0]
        st.markdown(
            f"""
            <div class="analytics-card">
                <h4 style="margin-top:0;">🎯 Best Match Assessment</h4>
                <p><strong>Top Job:</strong> {best_comp['jd_display']}</p>
                <p><strong>Match Score:</strong> <span style="font-size:18px; font-weight:700; color:#4F46E5;">{best_comp['match_score']:.2f}%</span></p>
                <p><strong>Status at {match_threshold:.0f}%:</strong> {'<span class="badge-qualifying">Qualified</span>' if best_comp['match_score'] >= match_threshold else '<span class="badge-unqualified">Below Threshold</span>'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Text Excerpt Preview
    with st.expander("📄 View Extracted Resume Text"):
        st.text_area("Extracted Text", r_text, height=180, disabled=True)

    # Ranking Chart for this candidate against all 8 JDs
    fig_r_all = px.bar(
        r_comps_all,
        x="match_score",
        y="jd_role",
        orientation="h",
        color="is_qualifying",
        color_discrete_map={True: "#10B981", False: "#6366F1"},
        labels={"match_score": "Match Score (%)", "jd_role": "Job Description Role"},
        title=f"All 8 JD Scores for {selected_r_code} (Threshold: {match_threshold:.0f}%)",
    )
    fig_r_all.add_vline(
        x=match_threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Threshold",
    )
    fig_r_all.update_layout(get_plotly_layout(f"JD Rankings for {selected_r_code}", is_dark))
    st.plotly_chart(fig_r_all, use_container_width=True)

    # Detailed Cards for Each JD
    st.markdown("#### Complete Ranking Breakdown Against All 8 JDs")
    for rank_idx, (_, comp_row) in enumerate(r_comps_all.iterrows(), start=1):
        is_qual = comp_row["is_qualifying"]
        status_tag = (
            '<span class="badge-qualifying">QUALIFYING MATCH</span>'
            if is_qual
            else '<span class="badge-unqualified">BELOW THRESHOLD</span>'
        )

        with st.expander(
            f"Rank #{rank_idx}: {comp_row['jd_display']} — {comp_row['match_score']:.2f}% ({'Qualifies' if is_qual else 'Below Threshold'})",
            expanded=(rank_idx == 1),
        ):
            c_score1, c_score2, c_score3, c_score4 = st.columns(4)
            c_score1.metric("Final Match Score", f"{comp_row['match_score']:.2f}%")
            c_score2.metric("TF-IDF Lexical (20%)", f"{comp_row['lexical_similarity']:.2f}%")
            c_score3.metric("Semantic Sim (50%)", f"{comp_row['semantic_similarity']:.2f}%")
            c_score4.metric("Skill Match (30%)", f"{comp_row['skill_coverage']:.2f}%")

            st.markdown(
                f"""
                <div style="margin: 10px 0;">
                    <strong>Qualification Status:</strong> {status_tag} &nbsp;|&nbsp;
                    <strong>Designed Role Match:</strong> {'✅ Yes (Target Role Family)' if comp_row['is_designed_match'] else 'ℹ️ Cross-Role Comparison'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_sk1, col_sk2 = st.columns(2)
            with col_sk1:
                st.markdown("**Matched / Overlapping Skills:**")
                st.markdown(
                    render_skill_chips_html(comp_row["matched_skills_list"], "matched"),
                    unsafe_allow_html=True,
                )
            with col_sk2:
                st.markdown("**Missing Required JD Skills:**")
                st.markdown(
                    render_skill_chips_html(comp_row["missing_skills_list"], "missing"),
                    unsafe_allow_html=True,
                )

# -------------------------------------------------------------
# PAGE 3: JD-WISE MATCHING
# -------------------------------------------------------------
elif page_selection == "💼 JD-Wise Matching":
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Job Description Shortlist & Candidate Rankings</div>
            <div class="page-desc">Inspect all 24 candidates ranked for any selected job description. Identifies qualifying applicants based on the active threshold and breaks down skill coverage.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jd_manifest = project_data["jd_manifest"]
    comp_df = dynamic_views["comparisons_df"]

    jd_options = [
        f"{row['jd_id']}: {row['role_title']} ({row['filename']})"
        for _, row in jd_manifest.iterrows()
    ]
    selected_jd_str = st.selectbox("Select Target Job Description", jd_options, index=0)
    selected_jd_code = selected_jd_str.split(":")[0].strip()

    jd_info = jd_manifest[jd_manifest["jd_id"] == selected_jd_code].iloc[0]
    jd_stem = Path(jd_info["filename"]).stem
    jd_doc = project_data["doc_text_map"].get(jd_stem, {})
    jd_text = jd_doc.get("text", "")

    # Filter all 24 candidates for this JD
    jd_candidates = comp_df[comp_df["jd_code"] == selected_jd_code].sort_values(
        "match_score", ascending=False
    ).reset_index(drop=True)

    qual_candidates = jd_candidates[jd_candidates["is_qualifying"]]

    # Summary Row
    c_jd1, c_jd2, c_jd3, c_jd4 = st.columns(4)
    c_jd1.markdown(
        render_kpi("Target Role", jd_info["role_title"], jd_info["filename"], "#4F46E5"),
        unsafe_allow_html=True,
    )
    c_jd2.markdown(
        render_kpi("Total Candidates", "24", "All Cohorts", "#7C3AED"),
        unsafe_allow_html=True,
    )
    c_jd3.markdown(
        render_kpi(
            "Qualifying Candidates",
            str(len(qual_candidates)),
            f"≥ {match_threshold:.0f}% Threshold",
            "#10B981",
        ),
        unsafe_allow_html=True,
    )
    c_jd4.markdown(
        render_kpi(
            "Top Candidate Score",
            f"{jd_candidates.iloc[0]['match_score']:.1f}%",
            jd_candidates.iloc[0]["resume_display"],
            "#0D9488",
        ),
        unsafe_allow_html=True,
    )

    with st.expander("📄 View Job Description Text & Requirements"):
        st.text_area("Extracted JD Content", jd_text, height=180, disabled=True)

    # Plotly candidate score ranking for this JD
    fig_jd_cands = px.bar(
        jd_candidates,
        x="resume_display",
        y="match_score",
        color="is_qualifying",
        color_discrete_map={True: "#10B981", False: "#6366F1"},
        labels={"resume_display": "Candidate", "match_score": "Match Score (%)"},
        title=f"All 24 Candidates Ranked for {jd_info['role_title']} (Threshold: {match_threshold:.0f}%)",
    )
    fig_jd_cands.add_hline(
        y=match_threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Threshold",
    )
    fig_jd_cands.update_layout(get_plotly_layout(f"Candidate Roster for {jd_info['role_title']}", is_dark))
    st.plotly_chart(fig_jd_cands, use_container_width=True)

    # Master Table of all 24 candidates
    st.markdown("#### Complete Candidate Roster (All 24 Resumes Ranked)")
    jd_table_view = jd_candidates[
        [
            "resume_code",
            "resume_filename",
            "resume_role",
            "match_score",
            "lexical_similarity",
            "semantic_similarity",
            "skill_coverage",
            "matched_skills",
            "missing_skills",
            "is_qualifying",
        ]
    ].rename(
        columns={
            "resume_code": "Resume ID",
            "resume_filename": "Filename",
            "resume_role": "Candidate Role",
            "match_score": "Score (%)",
            "lexical_similarity": "TF-IDF (%)",
            "semantic_similarity": "Semantic (%)",
            "skill_coverage": "Skill (%)",
            "matched_skills": "Matched Skills",
            "missing_skills": "Missing Skills",
            "is_qualifying": "Qualifies",
        }
    )

    st.dataframe(
        jd_table_view.style.format({
            "Score (%)": "{:.2f}%",
            "TF-IDF (%)": "{:.2f}%",
            "Semantic (%)": "{:.2f}%",
            "Skill (%)": "{:.2f}%",
        }),
        use_container_width=True,
    )

    # Download Shortlist CSV
    shortlist_csv = qual_candidates.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Shortlist for {jd_info['role_title']} ({len(qual_candidates)} Candidates)",
        data=shortlist_csv,
        file_name=f"shortlist_{jd_info['jd_id']}_{int(match_threshold)}.csv",
        mime="text/csv",
    )

# -------------------------------------------------------------
# PAGE 4: DETAILED MATCHING ANALYSIS
# -------------------------------------------------------------
elif page_selection == "🔍 Detailed Matching Analysis":
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Detailed Matching Analysis</div>
            <div class="page-desc">Inspect, filter, and compare all 192 pairwise resume–job description evaluations with side-by-side component breakdowns and exact formula calculations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_df = dynamic_views["comparisons_df"]

    # Filter Bar
    c_flt1, c_flt2, c_flt3, c_flt4 = st.columns(4)
    with c_flt1:
        res_role_flt = st.selectbox(
            "Resume Role",
            ["All Roles"] + project_data["all_roles"],
            key="det_res_role",
        )
    with c_flt2:
        jd_role_flt = st.selectbox(
            "JD Role",
            ["All Roles"] + project_data["all_roles"],
            key="det_jd_role",
        )
    with c_flt3:
        status_flt = st.selectbox(
            "Qualification Status",
            ["All Comparisons (192)", "Qualifying Matches Only", "Below Threshold Only"],
            key="det_status",
        )
    with c_flt4:
        score_min, score_max = st.slider(
            "Score Range Filter",
            min_value=0.0,
            max_value=100.0,
            value=(0.0, 100.0),
            step=5.0,
        )

    # Filter Data
    filtered_comps = comp_df.copy()
    if res_role_flt != "All Roles":
        filtered_comps = filtered_comps[filtered_comps["resume_role"] == res_role_flt]
    if jd_role_flt != "All Roles":
        filtered_comps = filtered_comps[filtered_comps["jd_role"] == jd_role_flt]
    if status_flt == "Qualifying Matches Only":
        filtered_comps = filtered_comps[filtered_comps["is_qualifying"]]
    elif status_flt == "Below Threshold Only":
        filtered_comps = filtered_comps[~filtered_comps["is_qualifying"]]
    filtered_comps = filtered_comps[
        (filtered_comps["match_score"] >= score_min) & (filtered_comps["match_score"] <= score_max)
    ]

    st.markdown(
        f"**Showing {len(filtered_comps)} of 192 comparisons** (Threshold: {match_threshold:.1f}%)"
    )

    table_cols = filtered_comps[
        [
            "resume_code",
            "resume_role",
            "jd_code",
            "jd_role",
            "match_score",
            "lexical_similarity",
            "semantic_similarity",
            "skill_coverage",
            "matched_skills",
            "missing_skills",
            "is_qualifying",
            "is_designed_match",
        ]
    ].rename(
        columns={
            "resume_code": "Resume ID",
            "resume_role": "Candidate Role",
            "jd_code": "JD ID",
            "jd_role": "Job Role",
            "match_score": "Score (%)",
            "lexical_similarity": "TF-IDF (%)",
            "semantic_similarity": "Semantic (%)",
            "skill_coverage": "Skill (%)",
            "matched_skills": "Matched Skills",
            "missing_skills": "Missing Skills",
            "is_qualifying": "Qualifying",
            "is_designed_match": "Designed Match",
        }
    )

    st.dataframe(
        table_cols.style.format({
            "Score (%)": "{:.2f}%",
            "TF-IDF (%)": "{:.2f}%",
            "Semantic (%)": "{:.2f}%",
            "Skill (%)": "{:.2f}%",
        }),
        use_container_width=True,
        height=320,
    )

    st.markdown("---")

    # Side-by-Side Deep Dive Comparator
    st.markdown("### 🔬 Pairwise Deep-Dive Inspector")
    col_sel_r, col_sel_j = st.columns(2)
    with col_sel_r:
        selected_r_id = st.selectbox(
            "Select Resume",
            project_data["all_resume_codes"],
            index=0,
            key="comp_r_select",
        )
    with col_sel_j:
        selected_j_id = st.selectbox(
            "Select Job Description",
            project_data["all_jd_codes"],
            index=0,
            key="comp_j_select",
        )

    pair_match = comp_df[
        (comp_df["resume_code"] == selected_r_id) & (comp_df["jd_code"] == selected_j_id)
    ].iloc[0]

    # Math Formula Callout
    score_math = (
        f"0.20 × {pair_match['lexical_similarity']:.2f}% (TF-IDF) + "
        f"0.50 × {pair_match['semantic_similarity']:.2f}% (Semantic) + "
        f"0.30 × {pair_match['skill_coverage']:.2f}% (Skill Coverage) = "
        f"**{pair_match['match_score']:.2f}%**"
    )

    st.markdown(
        f"""
        <div class="analytics-card">
            <h4 style="margin-top:0;">Formula Calculation: {pair_match['resume_code']} ↔ {pair_match['jd_code']}</h4>
            <p style="font-size:16px;"><code>{score_math}</code></p>
            <p><strong>Threshold Status:</strong> {'<span class="badge-qualifying">Qualifies (≥ ' + str(match_threshold) + '%)</span>' if pair_match['is_qualifying'] else '<span class="badge-unqualified">Below Threshold (< ' + str(match_threshold) + '%)</span>'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_side_r, col_side_j = st.columns(2)
    with col_side_r:
        st.markdown(f"#### 📄 Candidate: {pair_match['resume_display']}")
        r_text_preview = project_data["doc_text_map"].get(pair_match["resume_stem"], {}).get("text", "")
        st.text_area("Resume Excerpt", r_text_preview[:800] + "...", height=200, disabled=True)
        st.markdown("**Detected Skills in Resume:**")
        st.markdown(
            render_skill_chips_html(pair_match["matched_skills_list"], "matched"),
            unsafe_allow_html=True,
        )

    with col_side_j:
        st.markdown(f"#### 💼 Job: {pair_match['jd_display']}")
        j_text_preview = project_data["doc_text_map"].get(pair_match["jd_stem"], {}).get("text", "")
        st.text_area("JD Excerpt", j_text_preview[:800] + "...", height=200, disabled=True)
        st.markdown("**Missing JD Skills in Resume:**")
        st.markdown(
            render_skill_chips_html(pair_match["missing_skills_list"], "missing"),
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# PAGE 5: RESUME CHECKER / ATS READINESS
# -------------------------------------------------------------
elif page_selection == "⚡ Resume Checker / ATS Readiness":
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Resume Checker & Estimated ATS Readiness</div>
            <div class="page-desc">Upload a PDF resume to assess ATS readability, detect standard resume sections, analyze formatting quality, and calculate live compatibility against any of the 8 job descriptions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jd_manifest = project_data["jd_manifest"]
    jd_options = [
        f"{row['jd_id']}: {row['role_title']}"
        for _, row in jd_manifest.iterrows()
    ]

    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF format)",
            type=["pdf"],
            help="Uploaded documents are processed in memory and never permanently saved.",
        )
    with col_up2:
        target_jd_choice = st.selectbox(
            "Select Target Job Description to Compare Against",
            jd_options,
            index=0,
        )

    if uploaded_file is not None:
        target_jd_id = target_jd_choice.split(":")[0].strip()
        target_jd_row = jd_manifest[jd_manifest["jd_id"] == target_jd_id].iloc[0]
        target_jd_stem = Path(target_jd_row["filename"]).stem
        target_jd_text = project_data["doc_text_map"].get(target_jd_stem, {}).get("text", "")

        pdf_bytes = uploaded_file.read()

        with st.spinner("Extracting text and running ATS diagnostic checks..."):
            extracted_text, page_count, status = extract_text_from_pdf_bytes(pdf_bytes)

        if "ERROR" in status or "EMPTY_TEXT" in status:
            st.error(f"Text Extraction Issue: {status}")
            st.warning(
                "If this is an image-based or scanned document, standard text extractors cannot read it without OCR."
            )
        else:
            # Run section analysis and live scoring
            section_audit = analyze_sections(extracted_text)
            match_result = compute_live_match(extracted_text, target_jd_text)
            ats_audit = compute_ats_readiness_assessment(section_audit, match_result, page_count)

            st.success(f"Successfully processed **{uploaded_file.name}** ({page_count} pages, {len(extracted_text)} characters)")

            # Top KPI Summary
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    render_kpi(
                        "ATS Readiness Index",
                        f"{ats_audit['ats_score']:.0f}/100",
                        ats_audit["rating"],
                        ats_audit["rating_color"],
                    ),
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    render_kpi(
                        "Role Match Estimate",
                        f"{match_result['match_score']:.1f}%",
                        f"Against {target_jd_choice}",
                        "#4F46E5",
                    ),
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    render_kpi(
                        "Skill Coverage",
                        f"{match_result['skill_coverage']:.1f}%",
                        f"{len(match_result['matched_skills'])} of {len(match_result['jd_skills'])} Skills",
                        "#10B981",
                    ),
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    render_kpi(
                        "Document Quality",
                        f"{section_audit['word_count']} words",
                        f"{page_count} Page(s)",
                        "#0D9488",
                    ),
                    unsafe_allow_html=True,
                )

            # Score Decomposition
            st.markdown("#### 📊 Score Breakdown")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(
                    f"""
                    <div class="analytics-card">
                        <div class="kpi-label">Section Completeness</div>
                        <div class="kpi-value">{ats_audit['section_score']} / 40</div>
                        <div class="kpi-subtitle">Core headings & contact information</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc2:
                st.markdown(
                    f"""
                    <div class="analytics-card">
                        <div class="kpi-label">Parser & Format Quality</div>
                        <div class="kpi-value">{ats_audit['format_score']} / 25</div>
                        <div class="kpi-subtitle">Text density, length, and encoding</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with sc3:
                st.markdown(
                    f"""
                    <div class="analytics-card">
                        <div class="kpi-label">Role & Keyword Alignment</div>
                        <div class="kpi-value">{ats_audit['alignment_score']} / 35</div>
                        <div class="kpi-subtitle">Weighted composite against target JD</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Section Checklist Audit
            st.markdown("#### 📋 Section & Formatting Audit")
            col_sec1, col_sec2 = st.columns(2)

            with col_sec1:
                sec_dict = section_audit["sections"]
                st.markdown("**Essential Resume Sections Detected:**")
                st.markdown(f"- Contact Information: {'✅ Found' if section_audit['has_contact'] else '⚠️ Missing / Not Found'}")
                if section_audit.get("email"):
                    st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;Email: `{section_audit['email']}`")
                if section_audit.get("phone"):
                    st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;Phone: `{section_audit['phone']}`")
                st.markdown(f"- Professional Summary / Objective: {'✅ Found' if sec_dict.get('summary') else '⚠️ Missing / Incomplete'}")
                st.markdown(f"- Experience / Work History: {'✅ Found' if sec_dict.get('experience') else '⚠️ Missing / Incomplete'}")
                st.markdown(f"- Education / Qualifications: {'✅ Found' if sec_dict.get('education') else '⚠️ Missing / Incomplete'}")
                st.markdown(f"- Technical / Core Skills: {'✅ Found' if sec_dict.get('skills') else '⚠️ Missing / Incomplete'}")
                st.markdown(f"- Projects / Portfolio: {'✅ Found' if sec_dict.get('projects') else 'ℹ️ Optional / Not Detected'}")

            with col_sec2:
                st.markdown("**Formatting Diagnostics:**")
                st.markdown(f"- **Word Count:** {section_audit['word_count']} words")
                st.markdown(f"- **Character Count:** {section_audit['char_count']} characters")
                st.markdown(f"- **Page Count:** {page_count} page(s)")

                if section_audit["formatting_flags"]:
                    for flag in section_audit["formatting_flags"]:
                        st.warning(flag)
                else:
                    st.success("Clean font encoding and reasonable word density detected.")

            # Skill Gap Breakdown
            st.markdown("#### 🎯 Skill Gap Analysis for Target Role")
            col_sk1, col_sk2 = st.columns(2)
            with col_sk1:
                st.markdown(f"**Matched Skills ({len(match_result['matched_skills'])}):**")
                st.markdown(
                    render_skill_chips_html(match_result["matched_skills"], "matched"),
                    unsafe_allow_html=True,
                )
            with col_sk2:
                st.markdown(f"**Missing JD Skills ({len(match_result['missing_skills'])}):**")
                st.markdown(
                    render_skill_chips_html(match_result["missing_skills"], "missing"),
                    unsafe_allow_html=True,
                )

            # Concrete Recommendations
            st.markdown("#### 💡 Tailored Optimization Recommendations")
            for rec in ats_audit["recommendations"]:
                st.markdown(f"- 💡 {rec}")

            # Download Audit Summary
            audit_report_text = f"""RESUME ATS READINESS & ROLE COMPATIBILITY AUDIT
================================================
File: {uploaded_file.name}
Target Role: {target_jd_choice}
Date Evaluated: 2026-10-15

SCORES
------
Overall ATS Readiness: {ats_audit['ats_score']}/100 ({ats_audit['rating']})
- Section Completeness: {ats_audit['section_score']}/40
- Format & Parser Quality: {ats_audit['format_score']}/25
- Role Alignment: {ats_audit['alignment_score']}/35

Role Match Score: {match_result['match_score']:.2f}%
- TF-IDF Lexical Similarity: {match_result['lexical_similarity']:.2f}%
- Semantic Similarity: {match_result['semantic_similarity']:.2f}%
- Skill Coverage: {match_result['skill_coverage']:.2f}%

MATCHED SKILLS: {', '.join(match_result['matched_skills']) if match_result['matched_skills'] else 'None'}
MISSING JD SKILLS: {', '.join(match_result['missing_skills']) if match_result['missing_skills'] else 'None'}

RECOMMENDATIONS:
{chr(10).join(f"- {r}" for r in ats_audit['recommendations'])}

DISCLAIMER: This assessment is an algorithmic estimate for optimization purposes and does not guarantee passing proprietary employer ATS filters.
"""
            st.download_button(
                label="📥 Download ATS Audit Report (.txt)",
                data=audit_report_text.encode("utf-8"),
                file_name=f"ats_audit_{Path(uploaded_file.name).stem}.txt",
                mime="text/plain",
            )

    st.markdown(
        """
        <div class="custom-callout">
            <strong>ATS Disclaimer:</strong> This module produces an <em>estimated</em> ATS-readiness and role-match assessment based on text extractability, section heuristics, and semantic skill overlap. It is not a guarantee of passing any specific employer's proprietary ATS or recruitment screen.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# PAGE 6: REPORTS AND DOWNLOADS
# -------------------------------------------------------------
elif page_selection == "📥 Reports and Downloads":
    st.markdown(
        """
        <div class="page-header">
            <div class="page-title">Reports & Downloads Hub</div>
            <div class="page-desc">Generate and download comprehensive Excel reports dynamically formatted with the active match threshold, frozen headers, and auto-filters, or export individual CSV tables.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    resume_wise_df = dynamic_views["resume_wise_df"]
    jd_wise_df = dynamic_views["jd_wise_df"]
    comp_df = dynamic_views["comparisons_df"]

    # Big Excel Download Card
    st.markdown(
        f"""
        <div class="analytics-card">
            <h3 style="margin-top:0;">📊 Complete Multi-Sheet Excel Workbook</h3>
            <p>Generates an OpenPyXL workbook containing all <strong>7 required sheets</strong> dynamically calibrated to the <strong>{match_threshold:.1f}% threshold</strong>:</p>
            <ol>
                <li><code>Resume_Wise_Matches</code> (All 24 resumes, all qualifying JDs, scores, best match)</li>
                <li><code>JD_Wise_Matches</code> (All 8 JDs, all qualifying resumes, scores, top candidate)</li>
                <li><code>All_Comparisons</code> (All 192 pairwise comparisons with full score components)</li>
                <li><code>Best_Match_Per_Resume</code> (Highest-scoring JD for each candidate)</li>
                <li><code>Best_Match_Per_JD</code> (Top candidate for each job description)</li>
                <li><code>Skill_Gap_Analysis</code> (Detailed matched and missing skill breakdown per pair)</li>
                <li><code>Summary_Statistics</code> (Cohort metrics, threshold parameters, and evaluation benchmarks)</li>
            </ol>
            <p style="color:gray; font-size:12px;">Includes frozen headers, auto-filters, zebra striping, and percentage cell formatting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    excel_bytes = generate_excel_report(project_data, dynamic_views)
    st.download_button(
        label=f"📥 Download Complete Excel Workbook (Threshold: {match_threshold:.0f}%)",
        data=excel_bytes,
        file_name=f"resume_jd_matching_master_report_{int(match_threshold)}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📄 Individual CSV Exports")

    c_csv1, c_csv2, c_csv3 = st.columns(3)

    with c_csv1:
        st.markdown("**Resume-Wise Summary**")
        st.caption("All 24 resumes with qualifying matches.")
        r_csv = resume_wise_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Resume-Wise CSV",
            data=r_csv,
            file_name=f"resume_wise_matches_{int(match_threshold)}.csv",
            mime="text/csv",
        )

    with c_csv2:
        st.markdown("**JD-Wise Summary**")
        st.caption("All 8 JDs with qualifying candidates.")
        j_csv = jd_wise_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download JD-Wise CSV",
            data=j_csv,
            file_name=f"jd_wise_matches_{int(match_threshold)}.csv",
            mime="text/csv",
        )

    with c_csv3:
        st.markdown("**All 192 Comparisons**")
        st.caption("Full pairwise score matrix.")
        all_csv = comp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download All Comparisons CSV",
            data=all_csv,
            file_name="all_comparisons_192.csv",
            mime="text/csv",
        )
