import json
import os
import tempfile
import traceback
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

show_debug = globals().get("show_debug", False)


# ============================================================
# GLOBAL CLEAN VISUAL PATCH
# Auto-added by fix_budget_ui_beautiful_cards.py
# ============================================================

def apply_global_clean_visual_patch():
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1450px;
            padding-top: 1.4rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
            font-weight: 850 !important;
        }

        div[data-testid="stMetric"] {
            border-radius: 20px;
            padding: 1rem;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(148,163,184,0.18);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        .stButton > button {
            border-radius: 14px;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# END GLOBAL CLEAN VISUAL PATCH
# ============================================================

from pydub import AudioSegment
from pydub.silence import split_on_silence


# ============================================================
# FORCE SUPERVISOR-FRIENDLY UI EXTENSION
# Auto-added by force_supervisor_ui_upgrade.py
# ============================================================

def force_supervisor_ui_style():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.15), transparent 28%),
                radial-gradient(circle at top right, rgba(124, 58, 237, 0.14), transparent 30%),
                linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.035em;
            color: #0f172a;
        }

        .super-hero {
            padding: 2rem 2.2rem;
            border-radius: 30px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 50%, #6d28d9 100%);
            box-shadow: 0 25px 65px rgba(30, 64, 175, 0.28);
            color: white;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(255,255,255,0.18);
        }

        .super-hero-title {
            font-size: 2.45rem;
            font-weight: 900;
            line-height: 1.08;
            margin-bottom: 0.45rem;
        }

        .super-hero-subtitle {
            font-size: 1.03rem;
            line-height: 1.68;
            color: rgba(255,255,255,0.92);
            max-width: 1050px;
        }

        .super-pill-row {
            margin-top: 1.1rem;
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
        }

        .super-pill {
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.23);
            font-size: 0.86rem;
            font-weight: 750;
            color: white;
        }

        .super-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.35rem;
        }

        .super-card {
            padding: 1.05rem 1.1rem;
            border-radius: 22px;
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.07);
        }

        .super-card-kicker {
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }

        .super-card-title {
            color: #111827;
            font-size: 1.05rem;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }

        .super-card-text {
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .super-stat-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.7rem 0 1.3rem 0;
        }

        .super-stat {
            padding: 0.95rem;
            border-radius: 20px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }

        .super-stat-value {
            font-size: 1.25rem;
            font-weight: 900;
            color: #111827;
        }

        .super-stat-label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 700;
            margin-top: 0.1rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            padding: 1rem;
            border-radius: 20px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 800;
            color: #334155;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 900;
            color: #0f172a;
        }

        .stButton > button {
            border-radius: 15px;
            border: none;
            background: linear-gradient(135deg, #2563eb, #4f46e5);
            color: white;
            font-weight: 800;
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.24);
            padding: 0.7rem 1.1rem;
        }

        .stButton > button:hover {
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 18px 34px rgba(37, 99, 235, 0.32);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.76);
            padding: 0.6rem 1rem;
            font-weight: 750;
        }

        .stTabs [aria-selected="true"] {
            background: #2563eb !important;
            color: white !important;
        }

        .stExpander {
            border-radius: 18px !important;
            background: rgba(255,255,255,0.78) !important;
            border: 1px solid rgba(148, 163, 184, 0.25) !important;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.045);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }

        textarea {
            border-radius: 18px !important;
        }

        .super-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(79,70,229,0.5), transparent);
            margin: 1.2rem 0;
        }

        @media (max-width: 1000px) {
            .super-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .super-stat-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .super-hero-title {
                font-size: 1.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def force_supervisor_ui_header():
    st.markdown(
        """
        <div class="super-hero">
            <div class="super-hero-title">🎙️ Tamil Idea Pitch Evaluator</div>
            <div class="super-hero-subtitle">
                A machine-learning based Tamil audio/video idea pitch evaluation system that performs
                speech-to-text transcription, transcript refinement, confidence analysis, Tamil knowledge-base
                matching, budget feasibility estimation, and two-idea comparison.
            </div>
            <div class="super-pill-row">
                <span class="super-pill">Tamil STT / ASR</span>
                <span class="super-pill">ML Confidence Model</span>
                <span class="super-pill">Tamil Knowledge Base</span>
                <span class="super-pill">Budget Feasibility</span>
                <span class="super-pill">Audio + Video Input</span>
                <span class="super-pill">Idea Comparison</span>
            </div>
        </div>

        <div class="super-grid">
            <div class="super-card">
                <div class="super-card-kicker">Step 01</div>
                <div class="super-card-title">Upload Pitch</div>
                <div class="super-card-text">Upload Tamil audio/video. Audio is extracted and split into processable chunks.</div>
            </div>
            <div class="super-card">
                <div class="super-card-kicker">Step 02</div>
                <div class="super-card-title">Transcribe + Refine</div>
                <div class="super-card-text">Tamil speech is converted into raw and refined transcript for structured analysis.</div>
            </div>
            <div class="super-card">
                <div class="super-card-kicker">Step 03</div>
                <div class="super-card-title">Evaluate Knowledge</div>
                <div class="super-card-text">The pitch is compared with the Tamil-English CSE knowledge base using similarity metrics.</div>
            </div>
            <div class="super-card">
                <div class="super-card-kicker">Step 04</div>
                <div class="super-card-title">Score + Recommend</div>
                <div class="super-card-text">The system shows confidence, feasibility, budget estimate, feedback, and recommendation.</div>
            </div>
        </div>

        <div class="super-stat-strip">
            <div class="super-stat">
                <div class="super-stat-value">ASR</div>
                <div class="super-stat-label">Tamil Speech Recognition</div>
            </div>
            <div class="super-stat">
                <div class="super-stat-value">ML</div>
                <div class="super-stat-label">Confidence Prediction</div>
            </div>
            <div class="super-stat">
                <div class="super-stat-value">KB</div>
                <div class="super-stat-label">Knowledge Matching</div>
            </div>
            <div class="super-stat">
                <div class="super-stat-value">₹</div>
                <div class="super-stat-label">Budget Estimation</div>
            </div>
            <div class="super-stat">
                <div class="super-stat-value">2×</div>
                <div class="super-stat-label">Idea Comparison</div>
            </div>
        </div>

        <div class="super-divider"></div>
        """,
        unsafe_allow_html=True,
    )


def force_supervisor_sidebar():
    with st.sidebar:


        st.markdown("## 🎯 Project Dashboard")
        st.markdown("Tamil audio/video idea pitch evaluator.")
        st.markdown("---")
        st.markdown("### Core Modules")
        st.markdown("- **Tamil STT / ASR**")
        st.markdown("- **Transcript Refinement**")
        st.markdown("- **Chunk Analysis**")
        st.markdown("- **ML Confidence Model**")
        st.markdown("- **Knowledge-base Evaluation**")
        st.markdown("- **Budget Feasibility**")
        st.markdown("- **Two-Idea Comparison**")
        st.markdown("---")
        st.success("Deployment-safe mode enabled. Torch/Whisper option removed.")

# ============================================================
# END FORCE SUPERVISOR-FRIENDLY UI EXTENSION
# ============================================================


# ============================================================
# FINAL CLEAN BUDGET UI
# Auto-added by fix_budget_ui_final_clean.py
# ============================================================

def bf_render_budget_cards(budget_result):
    score = float(budget_result.get("budget_feasibility_out_of_10", 0.0))
    label = budget_result.get("budget_feasibility_label", "Not Available")
    estimation = budget_result.get("estimation", {})

    total_budget = estimation.get("total_estimated_budget", "Not Available")
    project_level = estimation.get("project_level", "Testing Prototype Level")

    st.markdown("## 💰 Budget Feasibility & Estimation")
    st.caption(
        "This section estimates whether the idea explains hardware cost, software/tool cost, "
        "deployment cost, maintenance cost, and overall affordability."
    )

    progress_value = max(0.0, min(1.0, score / 10.0))

    col1, col2, col3 = st.columns([1.1, 1.6, 1.8])

    with col1:
        with st.container():
            st.markdown("#### Budget Feasibility")
            st.markdown(f"### {score:.2f} / 10")
            st.progress(progress_value)

    with col2:
        with st.container():
            st.markdown("#### Budget Level")
            if score >= 7:
                st.success(label)
            elif score >= 4.5:
                st.warning(label)
            else:
                st.error(label)

    with col3:
        with st.container():
            st.markdown("#### Estimated Total Budget")
            st.markdown(f"### {total_budget}")

    st.info(f"**Project Scale:** {project_level}")

    st.caption(
        "Formula: 0.25×hardware cost clarity + 0.20×software/tool cost clarity "
        "+ 0.20×deployment cost clarity + 0.20×maintenance cost clarity "
        "+ 0.15×affordability/practicality"
    )


def bf_render_budget_breakdown_table(budget_result):
    estimation = budget_result.get("estimation", {})
    breakdown = budget_result.get("breakdown", {})

    budget_table_rows = [
        {
            "Budget Area": "Hardware",
            "Estimated Cost": estimation.get("hardware_estimate", "Not Available"),
            "Detected Status": estimation.get("hardware_label", "Not Available"),
            "Clarity Score": f"{bf_score_to_10(breakdown.get('hardware_cost_clarity', 0.0)):.2f} / 10",
        },
        {
            "Budget Area": "Software / Tools",
            "Estimated Cost": estimation.get("software_tool_estimate", "Not Available"),
            "Detected Status": estimation.get("software_tool_label", "Not Available"),
            "Clarity Score": f"{bf_score_to_10(breakdown.get('software_tool_cost_clarity', 0.0)):.2f} / 10",
        },
        {
            "Budget Area": "Deployment",
            "Estimated Cost": estimation.get("deployment_estimate", "Not Available"),
            "Detected Status": estimation.get("deployment_label", "Not Available"),
            "Clarity Score": f"{bf_score_to_10(breakdown.get('deployment_cost_clarity', 0.0)):.2f} / 10",
        },
        {
            "Budget Area": "Maintenance",
            "Estimated Cost": estimation.get("maintenance_estimate", "Not Available"),
            "Detected Status": estimation.get("maintenance_label", "Not Available"),
            "Clarity Score": f"{bf_score_to_10(breakdown.get('maintenance_cost_clarity', 0.0)):.2f} / 10",
        },
        {
            "Budget Area": "Affordability",
            "Estimated Cost": estimation.get("project_level", "Not Available"),
            "Detected Status": "Affordability / practicality signal",
            "Clarity Score": f"{bf_score_to_10(breakdown.get('affordability_practicality', 0.0)):.2f} / 10",
        },
    ]

    st.markdown("### 📊 Budget Breakdown")
    st.dataframe(
        budget_table_rows,
        use_container_width=True,
        hide_index=True,
    )


def bf_render_budget_feedback_box(feedback_items):
    st.markdown("### 💡 Budget Feedback")

    if not feedback_items:
        st.success("Budget explanation looks acceptable.")
        return

    for item in feedback_items:
        st.write(f"- {item}")


# ============================================================
# END FINAL CLEAN BUDGET UI
# ============================================================

# ============================================================
# BUDGET FEASIBILITY EXTENSION
# Auto-added by budget_feasibility_full_integrator.py
# ============================================================

def bf_clamp01(value):
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value > 10:
        value = value / 100.0
    elif value > 1:
        value = value / 10.0

    return max(0.0, min(1.0, value))


def bf_score_to_10(value):
    return round(bf_clamp01(value) * 10.0, 2)


def bf_count_hits(text, terms):
    text = (text or "").lower()
    return sum(1 for term in terms if term.lower() in text)


def bf_ratio_score(text, terms, max_hits):
    hits = bf_count_hits(text, terms)
    return max(0.0, min(1.0, hits / max_hits))


BF_TERMS = {
    "hardware": [
        "hardware", "sensor", "sensors", "camera", "cameras", "microphone",
        "raspberry pi", "arduino", "gpu", "server", "iot", "device",
        "storage device", "network device", "router", "scanner", "edge device",
        "ஹார்ட்வேர்", "சென்சார்", "கேமரா", "சாதனம்", "சேவையகம்",
        "கருவி", "மைக்ரோபோன்", "ஜிபியு",
    ],
    "no_hardware": [
        "no hardware", "without hardware", "software only", "software-only",
        "web app only", "website only", "no special hardware",
        "hardware தேவையில்லை", "சாப்ட்வேர் மட்டும்", "மென்பொருள் மட்டும்",
    ],
    "software_tools": [
        "software", "tool", "tools", "open source", "free tool", "library",
        "framework", "python", "java", "react", "angular", "node", "node.js",
        "django", "flask", "streamlit", "mysql", "mongodb", "postgresql",
        "api", "paid api", "free api", "license", "subscription",
        "scikit-learn", "tensorflow", "pytorch", "opencv",
        "சாப்ட்வேர்", "மென்பொருள்", "இலவச", "கருவி", "நூலகம்",
        "தரவுத்தளம்", "ஏபிஐ",
    ],
    "free_tools": [
        "free", "open source", "free api", "free tools", "no paid",
        "streamlit cloud", "local testing", "college server",
        "இலவச", "open-source", "குறைந்த செலவு",
    ],
    "paid_tools": [
        "paid", "subscription", "licensed", "premium", "paid api",
        "aws", "azure", "google cloud", "gpu", "server cost",
        "செலவு", "paid", "subscription",
    ],
    "deployment": [
        "deploy", "deployment", "hosting", "host", "server", "cloud",
        "streamlit cloud", "render", "railway", "vercel", "netlify",
        "aws", "azure", "google cloud", "docker", "local deployment",
        "college server", "production", "prototype",
        "டிப்ளாய்", "ஹோஸ்டிங்", "கிளவுட்", "சேவையகம்", "உள்ளூர்",
    ],
    "maintenance": [
        "maintenance", "maintain", "bug fixing", "update", "updates",
        "security update", "backup", "monitoring", "support",
        "model retraining", "dataset update", "server renewal",
        "api usage", "long term", "scaling", "logs",
        "பராமரிப்பு", "பிழை திருத்தம்", "புதுப்பிப்பு", "காப்புப்பிரதி",
        "கண்காணிப்பு", "ஆதரவு",
    ],
    "affordability": [
        "cost", "budget", "price", "cheap", "low cost", "affordable",
        "free", "open source", "student project", "testing purpose",
        "prototype", "minimal cost", "no hardware", "software only",
        "செலவு", "பட்ஜெட்", "குறைந்த செலவு", "இலவச", "மலிவு",
        "மாணவர் திட்டம்", "சோதனை", "prototype",
    ],
}


def bf_money_range(low, high):
    return f"₹{int(low):,} - ₹{int(high):,}"


def bf_detect_project_level(text):
    text_l = (text or "").lower()

    if any(term in text_l for term in ["enterprise", "company", "large scale", "production", "paid users"]):
        return "Production / Startup Level"

    if any(term in text_l for term in ["startup", "business", "customers", "scaling"]):
        return "Startup Prototype Level"

    if any(term in text_l for term in ["college", "student", "students", "testing purpose", "demo", "prototype"]):
        return "Student / College Prototype Level"

    return "Testing Prototype Level"


def bf_estimate_budget(transcript):
    text = transcript or ""
    text_l = text.lower()

    no_hardware = bf_count_hits(text, BF_TERMS["no_hardware"]) > 0
    hardware_hits = bf_count_hits(text, BF_TERMS["hardware"])
    paid_hits = bf_count_hits(text, BF_TERMS["paid_tools"])
    free_hits = bf_count_hits(text, BF_TERMS["free_tools"])
    deployment_hits = bf_count_hits(text, BF_TERMS["deployment"])
    maintenance_hits = bf_count_hits(text, BF_TERMS["maintenance"])

    project_level = bf_detect_project_level(text)

    # Hardware estimation
    if no_hardware:
        hardware_low, hardware_high = 0, 0
        hardware_label = "No special hardware required"
    elif hardware_hits >= 4:
        hardware_low, hardware_high = 5000, 50000
        hardware_label = "Hardware-heavy prototype"
    elif hardware_hits >= 1:
        hardware_low, hardware_high = 1000, 15000
        hardware_label = "Basic hardware/components may be required"
    else:
        hardware_low, hardware_high = 0, 5000
        hardware_label = "Hardware not clearly specified"

    # Software/tool estimation
    if paid_hits >= 2:
        software_low, software_high = 2000, 25000
        software_label = "May require paid APIs/tools/cloud services"
    elif free_hits >= 1 or any(tool in text_l for tool in ["python", "streamlit", "mysql", "mongodb", "react", "node"]):
        software_low, software_high = 0, 3000
        software_label = "Mostly free/open-source tools"
    else:
        software_low, software_high = 0, 8000
        software_label = "Software/tool cost not clearly specified"

    # Deployment estimation
    if any(term in text_l for term in ["streamlit cloud", "vercel", "netlify", "local deployment", "college server"]):
        deployment_low, deployment_high = 0, 2000
        deployment_label = "Free/low-cost testing deployment"
    elif any(term in text_l for term in ["aws", "azure", "google cloud", "production", "server"]):
        deployment_low, deployment_high = 5000, 30000
        deployment_label = "Paid cloud/server deployment may be required"
    elif deployment_hits >= 1:
        deployment_low, deployment_high = 1000, 10000
        deployment_label = "Deployment mentioned but cost not fully clear"
    else:
        deployment_low, deployment_high = 0, 5000
        deployment_label = "Deployment cost not specified"

    # Maintenance estimation
    if maintenance_hits >= 3:
        maintenance_low, maintenance_high = 2000, 15000
        maintenance_label = "Maintenance plan mentioned"
    elif maintenance_hits >= 1:
        maintenance_low, maintenance_high = 1000, 8000
        maintenance_label = "Basic maintenance mentioned"
    else:
        maintenance_low, maintenance_high = 0, 5000
        maintenance_label = "Maintenance cost not specified"

    total_low = hardware_low + software_low + deployment_low + maintenance_low
    total_high = hardware_high + software_high + deployment_high + maintenance_high

    return {
        "project_level": project_level,
        "hardware_estimate": bf_money_range(hardware_low, hardware_high),
        "hardware_label": hardware_label,
        "software_tool_estimate": bf_money_range(software_low, software_high),
        "software_tool_label": software_label,
        "deployment_estimate": bf_money_range(deployment_low, deployment_high),
        "deployment_label": deployment_label,
        "maintenance_estimate": bf_money_range(maintenance_low, maintenance_high),
        "maintenance_label": maintenance_label,
        "total_estimated_budget": bf_money_range(total_low, total_high),
    }


def bf_calculate_budget_feasibility(transcript, kb_result=None):
    text = transcript or ""

    hardware_score = bf_ratio_score(text, BF_TERMS["hardware"], 2)

    if bf_count_hits(text, BF_TERMS["no_hardware"]) > 0:
        hardware_score = max(hardware_score, 1.0)

    software_score = bf_ratio_score(text, BF_TERMS["software_tools"], 4)
    deployment_score = bf_ratio_score(text, BF_TERMS["deployment"], 3)
    maintenance_score = bf_ratio_score(text, BF_TERMS["maintenance"], 2)
    affordability_score = bf_ratio_score(text, BF_TERMS["affordability"], 3)

    # If user mentions free/open-source, improve affordability.
    if bf_count_hits(text, BF_TERMS["free_tools"]) > 0:
        affordability_score = max(affordability_score, 0.75)

    # Slight support from KB result if available.
    kb_support = 0.0
    if kb_result:
        try:
            kb_final = kb_result.get("final_score", 0.0)
            kb_support = bf_clamp01(kb_final) * 0.10
        except Exception:
            kb_support = 0.0

    score = (
        0.25 * hardware_score
        + 0.20 * software_score
        + 0.20 * deployment_score
        + 0.20 * maintenance_score
        + 0.15 * affordability_score
        + kb_support
    )

    score = max(0.0, min(1.0, score))

    if score >= 0.80:
        label = "Highly Feasible Budget"
    elif score >= 0.65:
        label = "Feasible Budget"
    elif score >= 0.45:
        label = "Moderate Budget Feasibility"
    elif score >= 0.25:
        label = "Weak Budget Feasibility"
    else:
        label = "Budget Not Clearly Explained"

    budget_estimation = bf_estimate_budget(text)

    return {
        "budget_feasibility_score": round(score, 4),
        "budget_feasibility_out_of_10": bf_score_to_10(score),
        "budget_feasibility_label": label,
        "formula": "0.25×hardware + 0.20×software/tools + 0.20×deployment + 0.20×maintenance + 0.15×affordability",
        "breakdown": {
            "hardware_cost_clarity": round(hardware_score, 4),
            "software_tool_cost_clarity": round(software_score, 4),
            "deployment_cost_clarity": round(deployment_score, 4),
            "maintenance_cost_clarity": round(maintenance_score, 4),
            "affordability_practicality": round(affordability_score, 4),
            "knowledge_base_support": round(kb_support, 4),
        },
        "estimation": budget_estimation,
    }


def bf_budget_feedback(budget_result):
    feedback = []
    score = budget_result.get("budget_feasibility_score", 0.0)
    breakdown = budget_result.get("breakdown", {})

    if score >= 0.75:
        feedback.append("Budget feasibility is strong. The pitch gives enough cost and practicality signals.")
    elif score >= 0.45:
        feedback.append("Budget feasibility is moderate. Add clearer cost details for tools, hosting, and maintenance.")
    else:
        feedback.append("Budget feasibility is weak. The pitch should clearly mention hardware cost, software/tool cost, deployment cost, maintenance cost, and affordability.")

    if breakdown.get("hardware_cost_clarity", 0.0) < 0.40:
        feedback.append("Mention whether hardware is required. If not required, clearly say it is a software-only project.")

    if breakdown.get("software_tool_cost_clarity", 0.0) < 0.40:
        feedback.append("Mention tools/libraries used, such as Python, Streamlit, React, Node.js, MySQL, MongoDB, or APIs.")

    if breakdown.get("deployment_cost_clarity", 0.0) < 0.40:
        feedback.append("Mention deployment plan, such as local testing, Streamlit Cloud, Render, Vercel, AWS, or college server.")

    if breakdown.get("maintenance_cost_clarity", 0.0) < 0.40:
        feedback.append("Mention maintenance plan, such as bug fixes, backup, updates, monitoring, or model retraining.")

    if breakdown.get("affordability_practicality", 0.0) < 0.40:
        feedback.append("Mention whether the project is low-cost, free-tool based, student-level, prototype-level, or startup-level.")

    return feedback

# ============================================================
# END BUDGET FEASIBILITY EXTENSION
# ============================================================


from speech_pipeline.inference_features import build_model_input
from semantic_eval.final_evaluator import TamilIdeaPitchEvaluator

from semantic_eval.tamil_transcript_refiner import (
    load_reference_texts,
    build_dataset_terms,
    refine_chunk_text,
    refine_full_transcript,
    analyze_chunk_with_dataset,
    format_raw_full_transcript,
    format_raw_transcript_by_chunks,
    format_refined_transcript_by_chunks,
    format_numbered_sentences,
)


MODEL_PATH = "models/confidence_model.pkl"
KB_PATH = "data/knowledge_base_tamil_reasoning.json"
TEXT_DATASET_PATH = "output/tamil_feature_dataset_labeled.csv"

SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
prefer_torch = False

SUPPORTED_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS + SUPPORTED_VIDEO_EXTENSIONS

UPLOAD_TYPES = [
    "wav", "mp3", "m4a", "flac", "ogg",
    "mp4", "mov", "mkv", "avi", "webm",
]

TARGET_CHUNK_MS = 15000
MIN_SILENCE_LEN = 700
SILENCE_THRESH_OFFSET_DB = 16
KEEP_SILENCE_MS = 250


def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return str(obj)


def _sequence_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def normalize_score(value):
    """
    Converts score to 0-1 range.
    Supports 0-1, 0-10, and 0-100 score formats.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value > 10:
        value = value / 100.0
    elif value > 1:
        value = value / 10.0

    return max(0.0, min(1.0, value))


def is_video_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def preview_uploaded_media(uploaded_file, label: str):
    """
    Shows uploaded file preview as video or audio based on extension.
    """
    if uploaded_file is None:
        return

    ext = Path(uploaded_file.name).suffix.lower()

    st.write(label)

    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)


def load_audio_file(media_path: str) -> AudioSegment:
    """
    Loads audio directly from audio files and extracts audio from video files.

    This keeps the rest of the pipeline unchanged:
    audio/video file -> AudioSegment -> chunking -> STT -> evaluation.

    FFmpeg must be installed for video extraction.
    """
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Media file not found: {media_path}")

    if not media_path.lower().endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported media format: {media_path}")

    try:
        audio = AudioSegment.from_file(media_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        return audio

    except Exception as e:
        raise RuntimeError(
            "Could not read audio from the uploaded file. "
            "If you uploaded a video, make sure it contains an audio track and FFmpeg is installed. "
            f"Original error: {e}"
        )


def split_long_chunk(chunk: AudioSegment, target_ms: int = TARGET_CHUNK_MS):
    """
    Splits long chunks into smaller fixed-size chunks.
    This prevents the last/long chunk from being skipped.
    """

    parts = []

    for start in range(0, len(chunk), target_ms):
        end = min(start + target_ms, len(chunk))
        part = chunk[start:end]

        if len(part) > 300:
            parts.append(part)

    return parts


def adaptive_split_audio(audio: AudioSegment):
    """
    Silence-aware splitting + fixed splitting fallback.
    Keeps the primary website behavior stable while processing full audio.
    """

    silence_thresh = (
        audio.dBFS - SILENCE_THRESH_OFFSET_DB
        if audio.dBFS != float("-inf")
        else -40
    )

    pieces = split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=silence_thresh,
        keep_silence=KEEP_SILENCE_MS,
    )

    if not pieces:
        return split_long_chunk(audio, TARGET_CHUNK_MS)

    final_chunks = []
    buffer = AudioSegment.silent(duration=0)

    for piece in pieces:
        if len(piece) > TARGET_CHUNK_MS:
            if len(buffer) > 0:
                final_chunks.extend(split_long_chunk(buffer, TARGET_CHUNK_MS))
                buffer = AudioSegment.silent(duration=0)

            final_chunks.extend(split_long_chunk(piece, TARGET_CHUNK_MS))
            continue

        if len(buffer) + len(piece) <= TARGET_CHUNK_MS:
            buffer += piece
        else:
            if len(buffer) > 0:
                final_chunks.extend(split_long_chunk(buffer, TARGET_CHUNK_MS))
            buffer = piece

    if len(buffer) > 0:
        final_chunks.extend(split_long_chunk(buffer, TARGET_CHUNK_MS))

    final_chunks = [chunk for chunk in final_chunks if len(chunk) > 300]

    return final_chunks


def predict_confidence(confidence_model, text: str):
    if not text.strip():
        return None

    feature_df = build_model_input(text, confidence_model)
    expected_features = getattr(confidence_model, "n_features_in_", None)

    if expected_features is not None and feature_df.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: model expects {expected_features} features, "
            f"but got {feature_df.shape[1]}. Current columns: {list(feature_df.columns)}"
        )

    prediction = confidence_model.predict(feature_df)[0]

    probability = None
    if hasattr(confidence_model, "predict_proba"):
        probability = float(max(confidence_model.predict_proba(feature_df)[0]))

    return prediction, probability


def confidence_reliability(
    text: str,
    prev_text: str | None,
    stt_quality: float,
    probability: float | None,
) -> float:
    length = len(text.split())
    length_score = min(length / 18.0, 1.0)

    repetition_penalty = _sequence_similarity(text, prev_text) if prev_text else 0.0
    model_prob = probability if probability is not None else 0.5

    score = (
        0.45 * stt_quality
        + 0.35 * model_prob
        + 0.20 * length_score
        - 0.25 * repetition_penalty
    )

    return round(max(0.0, min(1.0, score)), 4)


def aggregate_asr_quality(chunk_results):
    vals = [
        c["confidence_reliability"]
        for c in chunk_results
        if c.get("refined_text")
    ]

    if not vals:
        return 0.0

    return round(sum(vals) / len(vals), 4)


def calculate_audio_pitch_confidence(chunk_results, asr_quality):
    """
    Calculates how confidently the idea was pitched through audio using:
    ASR quality, chunk confidence, Tamil dataset coverage, technical strength,
    successful chunk ratio, and uncertainty penalty.
    """

    if not chunk_results:
        return {
            "audio_pitch_confidence_rate": 0.0,
            "audio_pitch_confidence_label": "No Audio Confidence",
            "audio_pitch_confidence_explanation": "No chunks were available for confidence analysis.",
            "audio_pitch_confidence_breakdown": {
                "asr_quality": 0.0,
                "avg_chunk_confidence": 0.0,
                "avg_dataset_coverage": 0.0,
                "technical_strength": 0.0,
                "successful_chunk_ratio": 0.0,
                "uncertainty_penalty": 0.0,
            },
        }

    total_chunks = len(chunk_results)

    successful_chunks = [
        row for row in chunk_results
        if str(row.get("raw_text", "")).strip()
    ]

    successful_chunk_ratio = len(successful_chunks) / total_chunks if total_chunks else 0.0

    confidence_values = [
        float(row.get("confidence_reliability", 0.0))
        for row in chunk_results
    ]

    dataset_coverage_values = [
        float(row.get("dataset_coverage", 0.0))
        for row in chunk_results
    ]

    technical_counts = [
        float(row.get("technical_word_count", 0.0))
        for row in chunk_results
    ]

    uncertain_counts = [
        float(row.get("uncertain_word_count", 0.0))
        for row in chunk_results
    ]

    avg_chunk_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )

    avg_dataset_coverage = (
        sum(dataset_coverage_values) / len(dataset_coverage_values)
        if dataset_coverage_values else 0.0
    )

    avg_technical_words = (
        sum(technical_counts) / len(technical_counts)
        if technical_counts else 0.0
    )

    avg_uncertain_words = (
        sum(uncertain_counts) / len(uncertain_counts)
        if uncertain_counts else 0.0
    )

    technical_strength = min(avg_technical_words / 4.0, 1.0)
    uncertainty_penalty = min(avg_uncertain_words / 3.0, 1.0)

    confidence_rate = (
        0.25 * float(asr_quality)
        + 0.25 * avg_chunk_confidence
        + 0.20 * avg_dataset_coverage
        + 0.15 * technical_strength
        + 0.15 * successful_chunk_ratio
        - 0.15 * uncertainty_penalty
    )

    confidence_rate = round(max(0.0, min(1.0, confidence_rate)), 4)

    if confidence_rate >= 0.80:
        label = "Very Confident Pitch"
        explanation = (
            "The audio pitch is clear, technically meaningful, strongly aligned with the Tamil dataset, "
            "and has high transcription reliability."
        )

    elif confidence_rate >= 0.65:
        label = "Confident Pitch"
        explanation = (
            "The audio pitch is mostly clear and meaningful, with good dataset coverage and reliable speech content."
        )

    elif confidence_rate >= 0.45:
        label = "Moderately Confident Pitch"
        explanation = (
            "The pitch has useful content, but it may need clearer explanation, stronger technical terms, "
            "or improved speech clarity."
        )

    elif confidence_rate >= 0.25:
        label = "Low Confidence Pitch"
        explanation = (
            "The pitch has weak clarity or limited dataset/technical coverage. The idea may need to be spoken "
            "more clearly with stronger reasoning."
        )

    else:
        label = "Very Low Confidence Pitch"
        explanation = (
            "The system could not confidently understand the pitch. This may be due to unclear audio, weak speech, "
            "low transcription success, or insufficient meaningful content."
        )

    return {
        "audio_pitch_confidence_rate": confidence_rate,
        "audio_pitch_confidence_label": label,
        "audio_pitch_confidence_explanation": explanation,
        "audio_pitch_confidence_breakdown": {
            "asr_quality": round(float(asr_quality), 4),
            "avg_chunk_confidence": round(avg_chunk_confidence, 4),
            "avg_dataset_coverage": round(avg_dataset_coverage, 4),
            "technical_strength": round(technical_strength, 4),
            "successful_chunk_ratio": round(successful_chunk_ratio, 4),
            "uncertainty_penalty": round(uncertainty_penalty, 4),
        },
    }


def generate_audio_based_feedback(chunk_results, audio_pitch_confidence, kb_result=None):
    """
    Generates dynamic feedback from actual audio/chunk analysis.
    """

    feedback = []

    if not chunk_results:
        return ["No valid audio chunks were available for feedback generation."]

    total_chunks = len(chunk_results)

    successful_chunks = [
        row for row in chunk_results
        if str(row.get("raw_text", "")).strip()
    ]

    weak_chunks = [
        row for row in chunk_results
        if row.get("analysis_status") in ["Weak", "Needs Review"]
    ]

    repeated_chunks = [
        row for row in chunk_results
        if float(row.get("repetition_score", 0.0)) > 0.70
    ]

    uncertain_chunks = [
        row for row in chunk_results
        if int(row.get("uncertain_word_count", 0)) > 0
    ]

    technical_chunks = [
        row for row in chunk_results
        if int(row.get("technical_word_count", 0)) > 0
    ]

    avg_stt_quality = sum(
        float(row.get("stt_quality", 0.0))
        for row in chunk_results
    ) / total_chunks

    avg_confidence = sum(
        float(row.get("confidence_reliability", 0.0))
        for row in chunk_results
    ) / total_chunks

    avg_dataset_coverage = sum(
        float(row.get("dataset_coverage", 0.0))
        for row in chunk_results
    ) / total_chunks

    successful_ratio = len(successful_chunks) / total_chunks

    pitch_rate = audio_pitch_confidence.get("audio_pitch_confidence_rate", 0.0)
    pitch_label = audio_pitch_confidence.get("audio_pitch_confidence_label", "Not Available")

    if pitch_rate >= 0.80:
        feedback.append(
            f"Your audio pitch sounds highly confident. The system classified it as '{pitch_label}' because most chunks were clear, meaningful, and aligned with the Tamil dataset."
        )
    elif pitch_rate >= 0.65:
        feedback.append(
            f"Your audio pitch is confident overall. The system classified it as '{pitch_label}', but a few areas can still be improved for stronger delivery."
        )
    elif pitch_rate >= 0.45:
        feedback.append(
            "Your audio pitch is moderately confident. The idea is understandable, but the delivery needs clearer explanation, stronger technical terms, or better speech clarity."
        )
    elif pitch_rate >= 0.25:
        feedback.append(
            "Your audio pitch has low confidence. The speech contains some useful content, but many chunks need clearer pronunciation, stronger reasoning, or better idea structure."
        )
    else:
        feedback.append(
            "The system could not confidently evaluate your audio pitch. This may be due to unclear speech, poor audio quality, low transcription success, or limited meaningful content."
        )

    if successful_ratio < 0.50:
        feedback.append(
            f"Only {len(successful_chunks)} out of {total_chunks} chunks produced recognizable text. Try speaking louder, closer to the microphone, and reduce background noise."
        )
    elif successful_ratio < 0.80:
        feedback.append(
            f"{len(successful_chunks)} out of {total_chunks} chunks were successfully transcribed. Some parts of the audio were unclear, so improving speech clarity will improve the evaluation."
        )
    else:
        feedback.append(
            f"Most chunks were successfully transcribed: {len(successful_chunks)} out of {total_chunks}. This means the audio was mostly understandable."
        )

    if avg_stt_quality < 0.45:
        feedback.append(
            "The average STT quality is low. The audio may contain unclear pronunciation, low volume, background noise, or long pauses."
        )
    elif avg_stt_quality < 0.65:
        feedback.append(
            "The STT quality is acceptable, but improving pronunciation and reducing noise can make the transcript more accurate."
        )
    else:
        feedback.append(
            "The STT quality is good, so the spoken content was reasonably clear for transcription."
        )

    if avg_dataset_coverage < 0.30:
        feedback.append(
            "The pitch has low alignment with the Tamil analyzed dataset. Add more relevant domain terms and explain the idea using clearer technical or project-related vocabulary."
        )
    elif avg_dataset_coverage < 0.60:
        feedback.append(
            "The pitch has moderate dataset alignment. It contains some relevant Tamil dataset terms, but adding more specific technical concepts will make it stronger."
        )
    else:
        feedback.append(
            "The pitch aligns well with the Tamil analyzed dataset, meaning the spoken content contains relevant idea-related terms."
        )

    if len(technical_chunks) == 0:
        feedback.append(
            "The pitch does not contain many technical words. Add details about components, working method, implementation, tools, or technology used."
        )
    elif len(technical_chunks) < total_chunks / 2:
        feedback.append(
            "Some chunks contain technical words, but the technical explanation is not consistent throughout the pitch. Try explaining the working process more clearly."
        )
    else:
        feedback.append(
            "The pitch includes technical terms across multiple chunks, which improves the strength of the idea presentation."
        )

    if uncertain_chunks:
        chunk_numbers = [str(row.get("chunk")) for row in uncertain_chunks[:5]]
        feedback.append(
            f"Uncertain wording was detected in chunk(s): {', '.join(chunk_numbers)}. Avoid vague phrases and speak with more direct explanation."
        )

    if repeated_chunks:
        chunk_numbers = [str(row.get("chunk")) for row in repeated_chunks[:5]]
        feedback.append(
            f"Repeated content was detected around chunk(s): {', '.join(chunk_numbers)}. Try reducing repeated statements and add new points such as benefits, implementation, cost, or real-world use."
        )

    if weak_chunks:
        chunk_numbers = [str(row.get("chunk")) for row in weak_chunks[:6]]
        feedback.append(
            f"Chunk(s) {', '.join(chunk_numbers)} need review because their clarity score or dataset match was low."
        )
    else:
        feedback.append(
            "No major weak chunks were detected. The pitch is structurally consistent across the audio."
        )

    if kb_result:
        try:
            scores = kb_result.get("scores", {})

            technical_score = normalize_score(scores.get("technical_correctness", 0.0))
            real_life_score = normalize_score(scores.get("real_life_probability", 0.0))
            theoretical_score = normalize_score(scores.get("theoretical_correctness", 0.0))

            if technical_score < 0.50:
                feedback.append(
                    "The technical correctness score is low. Explain how the idea works step-by-step and mention the required components or technology."
                )

            if real_life_score < 0.50:
                feedback.append(
                    "The real-life probability score is low. Add practical details such as where it can be used, who benefits, cost, limitations, and implementation feasibility."
                )

            if theoretical_score < 0.50:
                feedback.append(
                    "The theoretical correctness score is low. Improve the reasoning behind the idea and explain why the proposed method is valid."
                )

        except Exception:
            pass

    if avg_confidence >= 0.70 and avg_dataset_coverage >= 0.60:
        feedback.append(
            "Overall, your idea pitch is strong. To make it even better, add a short conclusion explaining impact, novelty, and practical implementation."
        )
    elif avg_confidence >= 0.50:
        feedback.append(
            "Overall, the pitch is understandable. Improve it by speaking more clearly, reducing vague words, and adding stronger technical and real-world explanation."
        )
    else:
        feedback.append(
            "Overall, the pitch needs improvement. Focus on clearer speech, better structure, stronger technical explanation, and more dataset-relevant terms."
        )

    return feedback


@st.cache_resource(show_spinner=False)
def load_cached_models(model_path: str, kb_path: str, text_dataset_path: str):
    confidence_model = joblib.load(model_path)
    evaluator = TamilIdeaPitchEvaluator(kb_path)

    reference_texts = load_reference_texts(
        text_dataset_path=text_dataset_path,
        kb_path=kb_path,
    )

    dataset_terms = build_dataset_terms(reference_texts)

    return confidence_model, evaluator, dataset_terms


def save_uploaded_audio(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def safe_transcribe_chunk(chunk, prefer_torch=False):
    """
    Keeps the STT interface and adds a final Streamlit-side retry.
    """

    from speech_pipeline.tamil_stt_free import transcribe_audiosegment_free

    try:
        transcribed = transcribe_audiosegment_free(
            chunk,
            prefer_torch=False,
        )
    except TypeError:
        transcribed = transcribe_audiosegment_free(chunk)

    if not isinstance(transcribed, dict):
        transcribed = {
            "text": "",
            "stt_engine": "unknown",
            "language_code": "ta-IN",
            "whisper_quality": 0.0,
            "error": "STT function did not return dictionary",
            "retry_attempt": None,
        }

    if not transcribed.get("text", "").strip():
        try:
            louder_chunk = chunk.apply_gain(8)

            try:
                retry_transcribed = transcribe_audiosegment_free(
                    louder_chunk,
                    prefer_torch=False,
                )
            except TypeError:
                retry_transcribed = transcribe_audiosegment_free(louder_chunk)

            if isinstance(retry_transcribed, dict) and retry_transcribed.get("text", "").strip():
                retry_transcribed["error"] = "Recovered after Streamlit-level volume retry"
                retry_transcribed["retry_attempt"] = retry_transcribed.get(
                    "retry_attempt",
                    "streamlit_plus_8db",
                )
                transcribed = retry_transcribed

        except Exception as retry_error:
            transcribed["error"] = (
                f"{transcribed.get('error')} | "
                f"Streamlit retry failed: {retry_error}"
            )

    transcribed.setdefault("text", "")
    transcribed.setdefault("stt_engine", "unknown")
    transcribed.setdefault("language_code", "ta-IN")
    transcribed.setdefault("whisper_quality", 0.0)
    transcribed.setdefault("error", None)
    transcribed.setdefault("retry_attempt", None)

    return transcribed


def run_pipeline(
    audio_path: str,
    model_path: str,
    kb_path: str,
    text_dataset_path: str,
    prefer_torch: bool = False,
):
    confidence_model, evaluator, dataset_terms = load_cached_models(
        model_path,
        kb_path,
        text_dataset_path,
    )

    audio = load_audio_file(audio_path)
    chunks = adaptive_split_audio(audio)

    if not chunks:
        raise ValueError("No valid audio chunks were created from the uploaded media.")

    raw_text_list = []
    refined_text_list = []
    chunk_results = []

    previous_refined_text = None
    total_chunks = len(chunks)

    progress_bar = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks, start=1):
        status.write(f"Processing chunk {i}/{total_chunks}...")

        transcribed = safe_transcribe_chunk(
            chunk,
            prefer_torch=False,
        )

        raw_text = transcribed.get("text", "").strip()
        refined_text = refine_chunk_text(raw_text)

        raw_text_list.append(raw_text)
        refined_text_list.append(refined_text)

        row = {
            "chunk": i,
            "total_chunks": total_chunks,
            "duration_sec": round(len(chunk) / 1000.0, 2),
            "raw_text": raw_text,
            "refined_text": refined_text,
            "stt_engine": transcribed.get("stt_engine"),
            "language_code": transcribed.get("language_code"),
            "stt_quality": transcribed.get("whisper_quality", 0.70),
            "stt_error": transcribed.get("error"),
            "stt_retry_attempt": transcribed.get("retry_attempt"),
            "prediction": None,
            "probability": None,
            "confidence_reliability": 0.0,
        }

        if refined_text.strip():
            pred_result = predict_confidence(confidence_model, refined_text)

            if pred_result is not None:
                row["prediction"], row["probability"] = pred_result

        row["confidence_reliability"] = confidence_reliability(
            text=refined_text,
            prev_text=previous_refined_text,
            stt_quality=row["stt_quality"],
            probability=row["probability"],
        )

        dataset_analysis = analyze_chunk_with_dataset(
            raw_text=raw_text,
            refined_text=refined_text,
            previous_refined_text=previous_refined_text,
            stt_quality=row["stt_quality"],
            model_probability=row["probability"],
            dataset_terms=dataset_terms,
        )

        row.update(dataset_analysis)

        previous_refined_text = refined_text
        chunk_results.append(row)

        progress_bar.progress(i / total_chunks)

    status.empty()
    progress_bar.empty()

    raw_full_transcript = " ".join(raw_text_list).strip()
    refined_full_input = " ".join(refined_text_list).strip()

    refined_full_transcript = refine_full_transcript(refined_full_input)

    neat_raw_full_transcript = format_raw_full_transcript(raw_full_transcript)

    raw_transcript_by_chunks = format_raw_transcript_by_chunks(chunk_results)
    refined_transcript_by_chunks = format_refined_transcript_by_chunks(chunk_results)
    numbered_refined_sentences = format_numbered_sentences(refined_full_transcript)

    asr_quality = aggregate_asr_quality(chunk_results)

    audio_pitch_confidence = calculate_audio_pitch_confidence(
        chunk_results=chunk_results,
        asr_quality=asr_quality,
    )

    kb_result = (
        evaluator.evaluate(refined_full_transcript, asr_quality=asr_quality)
        if refined_full_transcript
        else None
    )

    audio_based_feedback = generate_audio_based_feedback(
        chunk_results=chunk_results,
        audio_pitch_confidence=audio_pitch_confidence,
        kb_result=kb_result,
    )

    return {
        "raw_full_transcript": raw_full_transcript,
        "neat_raw_full_transcript": neat_raw_full_transcript,
        "refined_full_transcript": refined_full_transcript,
        "raw_transcript_by_chunks": raw_transcript_by_chunks,
        "refined_transcript_by_chunks": refined_transcript_by_chunks,
        "numbered_refined_sentences": numbered_refined_sentences,
        "asr_quality": asr_quality,
        "audio_pitch_confidence": audio_pitch_confidence,
        "audio_based_feedback": audio_based_feedback,
        "chunks": chunk_results,
        "kb_result": kb_result,
    }


def score_to_100(value):
    """
    Converts a score into 0-100.
    Supports 0-1, 0-10, and 0-100 formats.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value <= 1:
        value = value * 100
    elif value <= 10:
        value = value * 10

    return round(max(0.0, min(100.0, value)), 2)


def score_to_10(value):
    """
    Converts a score into 0-10.
    Supports 0-1, 0-10, and 0-100 formats.
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if value <= 1:
        value = value * 10
    elif value > 10:
        value = value / 10

    return round(max(0.0, min(10.0, value)), 2)


def format_score_out_of_total(value, total=10):
    return f"{score_to_10(value):.2f} / {total}"


def make_unique_key(*parts):
    cleaned = []
    for part in parts:
        part = str(part)
        part = part.replace(" ", "_").replace("/", "_").replace("-", "_")
        part = "".join(ch for ch in part if ch.isalnum() or ch == "_")
        cleaned.append(part[:50])
    return "_".join(cleaned)


def plot_interactive_bar_chart(title, labels, values, key_prefix, y_axis_mode="unit"):
    """
    Interactive bar chart with unique Streamlit key.
    y_axis_mode='unit' shows raw 0-1 values.
    y_axis_mode='percent' shows 0-100 values.
    """
    if y_axis_mode in ["score10", "percent"]:
        plot_values = [score_to_10(v) for v in values]
        y_title = "Score out of 10"
        y_range = [0, 10]
        text_values = [f"{v:.2f}" for v in plot_values]
        hover = "<b>%{x}</b><br>Score: %{y:.2f}/10<extra></extra>"
    else:
        plot_values = [normalize_score(v) for v in values]
        y_title = "Metric value"
        y_range = [0, 1]
        text_values = [f"{v:.4f}" for v in plot_values]
        hover = "<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>"

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=plot_values,
                text=text_values,
                textposition="auto",
                hovertemplate=hover,
            )
        ]
    )

    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="Metric",
        yaxis=dict(range=y_range),
        height=430,
        margin=dict(l=20, r=20, t=60, b=90),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=make_unique_key("bar", key_prefix, title),
    )


def display_match_metric_visuals(match, title, key_prefix):
    """
    Visualizes debug metric details like similarity, dense_similarity,
    sparse_similarity, and coverage using a modern hoverable table and bar graph.
    Pie charts are intentionally removed.
    """
    if not match:
        st.info("No match metrics available.")
        return

    metric_keys = [
        "similarity",
        "dense_similarity",
        "sparse_similarity",
        "coverage",
    ]

    rows = []
    labels = []
    values = []

    for key in metric_keys:
        if key in match:
            raw_value = float(match.get(key, 0.0))
            labels.append(key)
            values.append(raw_value)
            rows.append(
                {
                    "Metric": key,
                    "Raw Value": round(raw_value, 4),
                    "Score / 10": f"{score_to_10(raw_value):.2f} / 10",
                    "Level": score_interpretation(raw_value),
                }
            )

    if not rows:
        st.info("No metric values found for this match.")
        return

    st.markdown(f"**{title} — Metric Table**")

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Raw Value</b>", "<b>Score</b>", "<b>Level</b>"],
                    fill_color="#1f2937",
                    font=dict(color="white", size=14),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[
                        [row["Metric"] for row in rows],
                        [row["Raw Value"] for row in rows],
                        [row["Score / 10"] for row in rows],
                        [row["Level"] for row in rows],
                    ],
                    fill_color=["#111827", "#0f172a", "#111827", "#0f172a"],
                    font=dict(color="white", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"{title} — Hoverable Metric Table",
        height=250,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=make_unique_key("match_metric_table", key_prefix, title),
    )

    plot_interactive_bar_chart(
        title=f"{title} — Metric Bar Graph",
        labels=labels,
        values=values,
        key_prefix=key_prefix,
        y_axis_mode="unit",
    )


def display_all_match_metric_visuals(kb_result, prefix="kb"):
    if not kb_result:
        return

    matches_by_metric = kb_result.get("matches", {})

    if not matches_by_metric:
        st.info("No match details available for metric visualization.")
        return

    st.subheader("🧩 Debug Metric Graphs for Top Matches")

    for metric_index, (metric_name, matches) in enumerate(matches_by_metric.items(), start=1):
        with st.expander(f"{metric_name} - Match Metric Graphs", expanded=False):
            if not matches:
                st.write("No matches available.")
                continue

            for idx, match in enumerate(matches, start=1):
                st.markdown(f"### Match {idx}")

                display_match_metric_visuals(
                    match=match,
                    title=f"{metric_name} Match {idx}",
                    key_prefix=f"{prefix}_{metric_index}_{idx}",
                )

                matched_text = (
                    match.get("tamil")
                    or match.get("english")
                    or match.get("reasoning")
                    or ""
                )

                if matched_text:
                    with st.expander("Matched Knowledge-base Text", expanded=False):
                        st.write(matched_text)

                st.divider()


def score_interpretation(value):
    score = score_to_10(value)
    if score >= 8:
        return "Strong"
    if score >= 6:
        return "Good"
    if score >= 4:
        return "Moderate"
    if score >= 2:
        return "Weak"
    return "Very Weak"


def display_kb_score_table(kb_result, prefix="kb"):
    """Modern interactive Plotly table for knowledge-base scores, shown out of 10."""
    if not kb_result:
        return

    scores = kb_result.get("scores", {})

    rows = [
        ("Technical Correctness", scores.get("technical_correctness", 0.0)),
        ("Real-life Probability", scores.get("real_life_probability", 0.0)),
        ("Theoretical Correctness", scores.get("theoretical_correctness", 0.0)),
        ("Final Score", kb_result.get("final_score", 0.0)),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Score</b>", "<b>Level</b>"],
                    fill_color="#1f2937",
                    font=dict(color="white", size=14),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[
                        [r[0] for r in rows],
                        [format_score_out_of_total(r[1]) for r in rows],
                        [score_interpretation(r[1]) for r in rows],
                    ],
                    fill_color=["#111827", "#0f172a", "#111827"],
                    font=dict(color="white", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Knowledge-base Evaluation Table",
        height=270,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=make_unique_key("kb_table", prefix),
    )

def display_kb_score_visuals(kb_result, prefix="kb"):
    """
    Knowledge-base evaluation visualization.
    Scores are shown out of 10 to match the evaluator scale.
    """
    if not kb_result:
        st.info("No knowledge-base result available for visualization.")
        return

    scores = kb_result.get("scores", {})

    labels = [
        "Technical Correctness",
        "Real-life Probability",
        "Theoretical Correctness",
        "Final Score",
    ]

    values = [
        scores.get("technical_correctness", 0.0),
        scores.get("real_life_probability", 0.0),
        scores.get("theoretical_correctness", 0.0),
        kb_result.get("final_score", 0.0),
    ]

    display_kb_score_table(kb_result, prefix=prefix)

    plot_interactive_bar_chart(
        title="Knowledge-base Evaluation Scores",
        labels=labels,
        values=values,
        key_prefix=f"{prefix}_kb_scores_bar",
        y_axis_mode="score10",
    )


def display_chunk_cards(chunk_results, prefix="single"):
    st.subheader("Transcript for Each Chunk")

    chunk_view = st.radio(
        "Choose chunk transcript",
        ["Raw chunk transcript", "Refined chunk transcript"],
        horizontal=True,
        key=f"{prefix}_chunk_card_view_radio",
    )

    for row in chunk_results:
        chunk_no = row["chunk"]
        total_chunks = row["total_chunks"]
        duration = row["duration_sec"]

        if chunk_view == "Raw chunk transcript":
            text = row.get("raw_text", "")
        else:
            text = row.get("refined_text", "")

        text = str(text or "").strip()

        if not text:
            text = "[No clear speech detected in this chunk]"

        with st.expander(
            f"Chunk {chunk_no}/{total_chunks} | Duration: {duration}s",
            expanded=False,
        ):
            st.write(text)


def display_results(result, show_debug: bool, prefix="single"):
    raw_full_transcript = result.get("raw_full_transcript", "")
    neat_raw_full_transcript = result.get("neat_raw_full_transcript", "")
    refined_full_transcript = result.get("refined_full_transcript", "")
    asr_quality = result["asr_quality"]
    chunk_results = result["chunks"]
    kb_result = result["kb_result"]

    st.subheader("Statistical Overview")
    st.caption("Key transcript, chunk, ASR, confidence, budget, and knowledge-base indicators are summarized below.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Chunks Processed", len(chunk_results))
    c2.metric("ASR Quality", f"{asr_quality:.4f}")
    c3.metric("Transcript Words", len(refined_full_transcript.split()) if refined_full_transcript else 0)

    audio_pitch_confidence = result.get("audio_pitch_confidence", {})

    confidence_rate = audio_pitch_confidence.get("audio_pitch_confidence_rate", 0.0)
    confidence_label = audio_pitch_confidence.get(
        "audio_pitch_confidence_label",
        "Not Available",
    )
    confidence_explanation = audio_pitch_confidence.get(
        "audio_pitch_confidence_explanation",
        "Audio pitch confidence was not calculated.",
    )

    st.subheader("🎯 Audio Pitch Confidence Rate")

    p1, p2 = st.columns(2)

    p1.metric(
        "Audio Pitch Confidence",
        f"{confidence_rate * 100:.2f}%",
    )

    p2.metric(
        "Pitch Confidence Level",
        confidence_label,
    )

    st.info(confidence_explanation)

    with st.expander("Audio Pitch Confidence Breakdown", expanded=False):
        st.json(
            audio_pitch_confidence.get(
                "audio_pitch_confidence_breakdown",
                {},
            )
        )

    st.subheader("🗣️ Audio-Based Feedback")

    audio_based_feedback = result.get("audio_based_feedback", [])

    if audio_based_feedback:
        for index, item in enumerate(audio_based_feedback, start=1):
            st.write(f"{index}. {item}")
    else:
        st.write("No audio-based feedback was generated.")


    budget_result = bf_calculate_budget_feasibility(
        transcript=result.get("refined_full_transcript", ""),
        kb_result=kb_result,
    )

    bf_render_budget_cards(budget_result)
    bf_render_budget_breakdown_table(budget_result)
    bf_render_budget_feedback_box(bf_budget_feedback(budget_result))

    with st.expander("Budget Debug Details", expanded=False):
        st.json(budget_result)

    result["budget_feasibility"] = budget_result


    st.subheader("Complete Transcript View")

    transcript_view = st.radio(
        "Choose transcript view",
        [
            "Complete Raw Transcript",
            "Complete Refined Transcript",
            "Numbered Sentences / Key Points",
            "Raw Transcript by Chunks",
            "Refined Transcript by Chunks",
        ],
        horizontal=True,
        key=f"{prefix}_complete_transcript_view_radio",
    )

    if transcript_view == "Complete Raw Transcript":
        st.text_area(
            "Complete raw transcript from full audio",
            neat_raw_full_transcript or "[No raw transcript produced]",
            height=650,
            key=f"{prefix}_complete_raw_transcript_area",
        )

    elif transcript_view == "Complete Refined Transcript":
        st.text_area(
            "Complete refined Tamil transcript",
            refined_full_transcript or "[No refined transcript produced]",
            height=650,
            key=f"{prefix}_complete_refined_transcript_area",
        )

    elif transcript_view == "Numbered Sentences / Key Points":
        st.text_area(
            "Numbered sentences from refined transcript",
            result.get("numbered_refined_sentences", "") or "[No numbered sentences produced]",
            height=650,
            key=f"{prefix}_numbered_sentences_area",
        )

    elif transcript_view == "Raw Transcript by Chunks":
        st.text_area(
            "Raw transcript for every chunk",
            result.get("raw_transcript_by_chunks", "") or "[No raw chunk transcript produced]",
            height=750,
            key=f"{prefix}_raw_transcript_by_chunks_area",
        )

    elif transcript_view == "Refined Transcript by Chunks":
        st.text_area(
            "Refined transcript for every chunk",
            result.get("refined_transcript_by_chunks", "") or "[No refined chunk transcript produced]",
            height=750,
            key=f"{prefix}_refined_transcript_by_chunks_area",
        )

    display_chunk_cards(chunk_results, prefix=prefix)

    if show_debug:
        st.subheader("STT Status Check")

        st.dataframe(
            [
                {
                    "chunk": f"{row['chunk']}/{row['total_chunks']}",
                    "duration_sec": row["duration_sec"],
                    "raw_text_found": bool(str(row.get("raw_text", "")).strip()),
                    "raw_words": len(str(row.get("raw_text", "")).split()),
                    "stt_engine": row.get("stt_engine"),
                    "stt_quality": row.get("stt_quality"),
                    "retry_attempt": row.get("stt_retry_attempt"),
                    "stt_error": row.get("stt_error"),
                }
                for row in chunk_results
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Debug: Raw full transcript without formatting", expanded=False):
            st.text_area(
                "Raw full transcript",
                raw_full_transcript or "[No raw transcript produced]",
                height=300,
                key=f"{prefix}_debug_raw_full_transcript_area",
            )

    st.subheader("Chunk-level Analysis")

    st.dataframe(
        [
            {
                "chunk": f"{row['chunk']}/{row['total_chunks']}",
                "duration_sec": row["duration_sec"],
                "status": row["analysis_status"],
                "clarity_score": row["clarity_score"],
                "audio_pitching_signal": row.get("confidence_reliability"),
                "confidence_reliability": row["confidence_reliability"],
                "prediction": row["prediction"],
                "model_probability": row["probability"],
                "stt_engine": row["stt_engine"],
                "stt_quality": row["stt_quality"],
                "dataset_coverage": row["dataset_coverage"],
                "technical_words": row["technical_word_count"],
                "uncertain_words": row["uncertain_word_count"],
                "repetition_score": row["repetition_score"],
                "issues": row["analysis_issues"],
                "refined_text": row["refined_text"],
            }
            for row in chunk_results
        ],
        use_container_width=True,
        hide_index=True,
    )

    if show_debug:
        st.subheader("Detailed Chunk Debug")

        st.dataframe(
            [
                {
                    "chunk": f"{row['chunk']}/{row['total_chunks']}",
                    "duration_sec": row["duration_sec"],
                    "raw_text": row["raw_text"],
                    "refined_text": row["refined_text"],
                    "matched_dataset_terms": ", ".join(row.get("dataset_matched_terms", [])),
                    "stt_error": row.get("stt_error"),
                    "retry_attempt": row.get("stt_retry_attempt"),
                }
                for row in chunk_results
            ],
            use_container_width=True,
            hide_index=True,
        )


    if kb_result:
        st.subheader("Knowledge-base Evaluation")

        scores = kb_result["scores"]
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Technical Correctness", format_score_out_of_total(scores.get('technical_correctness', 0.0)))
        m2.metric("Real-life Probability", format_score_out_of_total(scores.get('real_life_probability', 0.0)))
        m3.metric("Theoretical Correctness", format_score_out_of_total(scores.get('theoretical_correctness', 0.0)))
        m4.metric("Final Score", format_score_out_of_total(kb_result.get("final_score", 0.0)))

        display_kb_score_visuals(kb_result, prefix=prefix)

        st.subheader("Explanations")
        for key, value in kb_result["explanations"].items():
            with st.expander(key, expanded=True):
                st.write(value)

        st.subheader("Knowledge-base Feedback")
        for item in kb_result["feedback"]:
            st.write(f"- {item}")

        display_all_match_metric_visuals(kb_result, prefix=prefix)

        if show_debug and "debug" in kb_result:
            st.subheader("Debug Details")
            st.json(kb_result["debug"])

    export_payload = {
        "raw_full_transcript": result.get("raw_full_transcript", ""),
        "neat_raw_full_transcript": result.get("neat_raw_full_transcript", ""),
        "refined_full_transcript": result.get("refined_full_transcript", ""),
        "numbered_refined_sentences": result.get("numbered_refined_sentences", ""),
        "raw_transcript_by_chunks": result.get("raw_transcript_by_chunks", ""),
        "refined_transcript_by_chunks": result.get("refined_transcript_by_chunks", ""),
        "asr_quality": asr_quality,
        "audio_pitch_confidence": result.get("audio_pitch_confidence", {}),
        "audio_based_feedback": result.get("audio_based_feedback", []),
        "budget_feasibility": result.get("budget_feasibility", {}),
        "chunks": chunk_results,
        "kb_result": kb_result,
    }

    st.download_button(
        "Download evaluation JSON",
        data=json.dumps(
            export_payload,
            ensure_ascii=False,
            indent=2,
            default=convert_numpy,
        ),
        file_name=f"{prefix}_tamil_pitch_evaluation.json",
        mime="application/json",
        use_container_width=True,
        key=f"{prefix}_download_evaluation_json_button",
    )

def get_safe_score(kb_result, key, default=0.0):
    try:
        return normalize_score(kb_result["scores"].get(key, default))
    except Exception:
        return default


def calculate_favourability_score(result):
    """
    Calculates a final score to decide which idea is more favourable.
    Keeps the chart meaningful without flooding every metric as /100 text.
    """

    kb_result = result.get("kb_result")

    if not kb_result:
        technical = 0.0
        real_life = 0.0
        theoretical = 0.0
        kb_final = 0.0
    else:
        technical = get_safe_score(kb_result, "technical_correctness")
        real_life = get_safe_score(kb_result, "real_life_probability")
        theoretical = get_safe_score(kb_result, "theoretical_correctness")
        kb_final = normalize_score(kb_result.get("final_score", 0.0))

    audio_pitch_confidence = normalize_score(
        result.get("audio_pitch_confidence", {}).get("audio_pitch_confidence_rate", 0.0)
    )

    asr_quality = normalize_score(result.get("asr_quality", 0.0))

    chunk_results = result.get("chunks", [])

    avg_dataset_coverage = 0.0
    avg_clarity = 0.0
    technical_strength = 0.0
    successful_chunk_ratio = 0.0

    if chunk_results:
        successful_chunks = [
            row for row in chunk_results
            if str(row.get("raw_text", "")).strip()
        ]

        successful_chunk_ratio = len(successful_chunks) / len(chunk_results)

        avg_dataset_coverage = sum(
            float(row.get("dataset_coverage", 0.0))
            for row in chunk_results
        ) / len(chunk_results)

        avg_clarity = sum(
            float(row.get("clarity_score", 0.0))
            for row in chunk_results
        ) / len(chunk_results)

        avg_technical_words = sum(
            float(row.get("technical_word_count", 0.0))
            for row in chunk_results
        ) / len(chunk_results)

        technical_strength = min(avg_technical_words / 4.0, 1.0)

    favourability_score = (
        0.24 * technical
        + 0.20 * real_life
        + 0.12 * theoretical
        + 0.14 * kb_final
        + 0.10 * audio_pitch_confidence
        + 0.07 * asr_quality
        + 0.05 * avg_dataset_coverage
        + 0.04 * avg_clarity
        + 0.02 * technical_strength
        + 0.02 * successful_chunk_ratio
    )

    return {
        "technical_correctness": round(technical, 4),
        "real_life_probability": round(real_life, 4),
        "theoretical_correctness": round(theoretical, 4),
        "knowledge_base_final_score": round(kb_final, 4),
        "audio_pitch_confidence": round(audio_pitch_confidence, 4),
        "asr_quality": round(asr_quality, 4),
        "dataset_coverage": round(avg_dataset_coverage, 4),
        "chunk_clarity": round(avg_clarity, 4),
        "technical_strength": round(technical_strength, 4),
        "successful_chunk_ratio": round(successful_chunk_ratio, 4),
        "favourability_score": round(favourability_score, 4),
    }


def decide_better_idea(score_1, score_2):
    diff = score_1["favourability_score"] - score_2["favourability_score"]

    if abs(diff) < 0.05:
        return {
            "winner": "Both ideas are very close",
            "winner_key": "tie",
            "reason": (
                "Both ideas have similar favourability scores. Choose based on novelty, cost, "
                "implementation difficulty, required resources, and expected real-world impact."
            ),
        }

    if diff > 0:
        return {
            "winner": "Idea 1 is more favourable",
            "winner_key": "idea_1",
            "reason": (
                "Idea 1 has a stronger overall score across technical correctness, real-life feasibility, "
                "knowledge-base score, audio pitch confidence, dataset coverage, and chunk clarity."
            ),
        }

    return {
        "winner": "Idea 2 is more favourable",
        "winner_key": "idea_2",
        "reason": (
            "Idea 2 has a stronger overall score across technical correctness, real-life feasibility, "
            "knowledge-base score, audio pitch confidence, dataset coverage, and chunk clarity."
        ),
    }


def generate_recommendation_text(result_1, result_2, score_1, score_2, decision):
    if decision["winner_key"] == "tie":
        return (
            "Both ideas are close. The system cannot strongly prefer one idea from the current audio and knowledge-base analysis. "
            "Choose the idea that is cheaper, easier to implement, more novel, and has better real-world usefulness."
        )

    selected = "idea_1" if decision["winner_key"] == "idea_1" else "idea_2"
    other = "idea_2" if selected == "idea_1" else "idea_1"

    selected_score = score_1 if selected == "idea_1" else score_2
    other_score = score_2 if selected == "idea_1" else score_1

    selected_name = "Idea 1" if selected == "idea_1" else "Idea 2"
    other_name = "Idea 2" if selected == "idea_1" else "Idea 1"

    strengths = []

    metric_names = {
        "technical_correctness": "technical correctness",
        "real_life_probability": "real-life feasibility",
        "theoretical_correctness": "theoretical correctness",
        "knowledge_base_final_score": "knowledge-base score",
        "audio_pitch_confidence": "audio pitch confidence",
        "asr_quality": "speech clarity",
        "dataset_coverage": "Tamil dataset alignment",
        "chunk_clarity": "chunk clarity",
        "technical_strength": "technical vocabulary strength",
        "successful_chunk_ratio": "successful transcription ratio",
    }

    for key, label in metric_names.items():
        if selected_score.get(key, 0.0) > other_score.get(key, 0.0):
            strengths.append(label)

    top_strengths = strengths[:4]
    strength_text = ", ".join(top_strengths) if top_strengths else "overall evaluation balance"

    return (
        f"{selected_name} is recommended because it performs better in {strength_text}. "
        f"It has a favourability score of {format_score_out_of_total(selected_score['favourability_score'])}, "
        f"while {other_name} has {format_score_out_of_total(other_score['favourability_score'])}. "
        f"To improve {other_name}, strengthen the technical explanation, real-life feasibility, dataset-related terms, and audio delivery clarity."
    )


def generate_dynamic_comparison_feedback(result_1, result_2, score_1, score_2, decision):
    feedback = []

    diff = round(abs(score_1["favourability_score"] - score_2["favourability_score"]), 4)

    feedback.append(
        f"The favourability score difference is {score_to_10(diff):.2f} out of 10. Result: {decision['winner']}."
    )

    metrics = [
        ("technical_correctness", "technical correctness"),
        ("real_life_probability", "real-life feasibility"),
        ("theoretical_correctness", "theoretical correctness"),
        ("knowledge_base_final_score", "knowledge-base final score"),
        ("audio_pitch_confidence", "audio pitch confidence"),
        ("asr_quality", "speech/transcription clarity"),
        ("dataset_coverage", "Tamil dataset alignment"),
        ("chunk_clarity", "chunk clarity"),
        ("technical_strength", "technical vocabulary strength"),
        ("successful_chunk_ratio", "successful transcription ratio"),
    ]

    for key, label in metrics:
        value_1 = score_1.get(key, 0.0)
        value_2 = score_2.get(key, 0.0)
        gap = abs(value_1 - value_2)

        if gap < 0.03:
            feedback.append(f"Both ideas are almost equal in {label}.")
        elif value_1 > value_2:
            feedback.append(f"Idea 1 is stronger in {label} by {score_to_10(gap):.2f} out of 10.")
        else:
            feedback.append(f"Idea 2 is stronger in {label} by {score_to_10(gap):.2f} out of 10.")

    idea_1_words = len(result_1.get("refined_full_transcript", "").split())
    idea_2_words = len(result_2.get("refined_full_transcript", "").split())

    if idea_1_words > idea_2_words:
        feedback.append(
            f"Idea 1 has a more detailed refined transcript with {idea_1_words} words compared to Idea 2 with {idea_2_words} words."
        )
    elif idea_2_words > idea_1_words:
        feedback.append(
            f"Idea 2 has a more detailed refined transcript with {idea_2_words} words compared to Idea 1 with {idea_1_words} words."
        )
    else:
        feedback.append("Both ideas have the same refined transcript length.")

    idea_1_feedback = result_1.get("audio_based_feedback", [])
    idea_2_feedback = result_2.get("audio_based_feedback", [])

    if idea_1_feedback:
        feedback.append(f"Idea 1 audio analysis note: {idea_1_feedback[0]}")

    if idea_2_feedback:
        feedback.append(f"Idea 2 audio analysis note: {idea_2_feedback[0]}")

    feedback.append(generate_recommendation_text(result_1, result_2, score_1, score_2, decision))

    return feedback


def display_favourability_contribution_table(score_1, score_2):
    """
    Shows Idea 1 and Idea 2 favourability contribution as a modern interactive table.
    Pie charts are intentionally removed.
    """
    contribution_rows = [
        ("Technical", score_1["technical_correctness"] * 0.24, score_2["technical_correctness"] * 0.24),
        ("Real-life", score_1["real_life_probability"] * 0.20, score_2["real_life_probability"] * 0.20),
        ("Theory", score_1["theoretical_correctness"] * 0.12, score_2["theoretical_correctness"] * 0.12),
        ("KB Final", score_1["knowledge_base_final_score"] * 0.14, score_2["knowledge_base_final_score"] * 0.14),
        ("Audio Confidence", score_1["audio_pitch_confidence"] * 0.10, score_2["audio_pitch_confidence"] * 0.10),
        ("ASR", score_1["asr_quality"] * 0.07, score_2["asr_quality"] * 0.07),
        ("Dataset", score_1["dataset_coverage"] * 0.05, score_2["dataset_coverage"] * 0.05),
        ("Chunk Clarity", score_1["chunk_clarity"] * 0.04, score_2["chunk_clarity"] * 0.04),
        ("Technical Strength", score_1["technical_strength"] * 0.02, score_2["technical_strength"] * 0.02),
        ("Successful Chunks", score_1["successful_chunk_ratio"] * 0.02, score_2["successful_chunk_ratio"] * 0.02),
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "<b>Contribution Factor</b>",
                        "<b>Idea 1 Contribution</b>",
                        "<b>Idea 2 Contribution</b>",
                        "<b>Stronger</b>",
                    ],
                    fill_color="#1f2937",
                    font=dict(color="white", size=14),
                    align="left",
                    height=34,
                ),
                cells=dict(
                    values=[
                        [row[0] for row in contribution_rows],
                        [f"{score_to_10(row[1]):.2f} / 10" for row in contribution_rows],
                        [f"{score_to_10(row[2]):.2f} / 10" for row in contribution_rows],
                        [
                            "Idea 1" if row[1] > row[2] else "Idea 2" if row[2] > row[1] else "Equal"
                            for row in contribution_rows
                        ],
                    ],
                    fill_color=["#111827", "#0f172a", "#111827", "#0f172a"],
                    font=dict(color="white", size=13),
                    align="left",
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Favourability Contribution Table",
        height=430,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="favourability_contribution_table_unique",
    )


def display_comparison_visuals(score_1, score_2):
    """
    Keeps the grouped comparison chart style and fixes duplicate Plotly IDs using a key.
    """
    st.subheader("📊 Visual Comparison")

    labels = [
        "Technical Correctness",
        "Real-life Probability",
        "Theoretical Correctness",
        "KB Final Score",
        "Audio Pitch Confidence",
        "ASR Quality",
        "Dataset Coverage",
        "Chunk Clarity",
        "Technical Strength",
        "Successful Chunks",
        "Favourability Score",
    ]

    idea_1_values = [
        score_1["technical_correctness"],
        score_1["real_life_probability"],
        score_1["theoretical_correctness"],
        score_1["knowledge_base_final_score"],
        score_1["audio_pitch_confidence"],
        score_1["asr_quality"],
        score_1["dataset_coverage"],
        score_1["chunk_clarity"],
        score_1["technical_strength"],
        score_1["successful_chunk_ratio"],
        score_1["favourability_score"],
    ]

    idea_2_values = [
        score_2["technical_correctness"],
        score_2["real_life_probability"],
        score_2["theoretical_correctness"],
        score_2["knowledge_base_final_score"],
        score_2["audio_pitch_confidence"],
        score_2["asr_quality"],
        score_2["dataset_coverage"],
        score_2["chunk_clarity"],
        score_2["technical_strength"],
        score_2["successful_chunk_ratio"],
        score_2["favourability_score"],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[score_to_10(v) for v in idea_1_values],
            name="Idea 1",
            hovertemplate="<b>%{x}</b><br>Idea 1: %{y:.2f}/10<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[score_to_10(v) for v in idea_2_values],
            name="Idea 2",
            hovertemplate="<b>%{x}</b><br>Idea 2: %{y:.2f}/10<extra></extra>",
        )
    )

    fig.update_layout(
        title="Idea 1 vs Idea 2 Score Comparison",
        yaxis_title="Score out of 10",
        xaxis_title="Metric",
        yaxis=dict(range=[0, 10]),
        barmode="group",
        height=560,
        margin=dict(l=20, r=20, t=60, b=140),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="comparison_grouped_score_chart_unique",
    )

    st.subheader("📋 Favourability Contribution Table")
    display_favourability_contribution_table(score_1, score_2)


def display_comparison_results(result_1, result_2, show_debug: bool):
    score_1 = calculate_favourability_score(result_1)
    score_2 = calculate_favourability_score(result_2)

    decision = decide_better_idea(score_1, score_2)

    comparison_feedback = generate_dynamic_comparison_feedback(
        result_1=result_1,
        result_2=result_2,
        score_1=score_1,
        score_2=score_2,
        decision=decision,
    )

    st.header("⚖️ Two Idea Comparison Result")

    score_col_1, score_col_2 = st.columns(2)

    score_col_1.metric(
        "Idea 1 Favourability Score",
        format_score_out_of_total(score_1["favourability_score"]),
    )

    score_col_2.metric(
        "Idea 2 Favourability Score",
        format_score_out_of_total(score_2["favourability_score"]),
    )

    if decision["winner_key"] == "tie":
        st.warning(decision["winner"])
    else:
        st.success(decision["winner"])

    st.write(decision["reason"])

    recommendation = generate_recommendation_text(
        result_1=result_1,
        result_2=result_2,
        score_1=score_1,
        score_2=score_2,
        decision=decision,
    )

    st.subheader("✅ Final Recommendation")
    st.info(recommendation)

    st.subheader("Comparison Scores")

    comparison_rows = [
        {"Metric": "Technical Correctness", "Idea 1": format_score_out_of_total(score_1['technical_correctness']), "Idea 2": format_score_out_of_total(score_2['technical_correctness'])},
        {"Metric": "Real-life Probability", "Idea 1": format_score_out_of_total(score_1['real_life_probability']), "Idea 2": format_score_out_of_total(score_2['real_life_probability'])},
        {"Metric": "Theoretical Correctness", "Idea 1": format_score_out_of_total(score_1['theoretical_correctness']), "Idea 2": format_score_out_of_total(score_2['theoretical_correctness'])},
        {"Metric": "Knowledge-base Final Score", "Idea 1": format_score_out_of_total(score_1['knowledge_base_final_score']), "Idea 2": format_score_out_of_total(score_2['knowledge_base_final_score'])},
        {"Metric": "Audio Pitch Confidence", "Idea 1": format_score_out_of_total(score_1['audio_pitch_confidence']), "Idea 2": format_score_out_of_total(score_2['audio_pitch_confidence'])},
        {"Metric": "ASR Quality", "Idea 1": format_score_out_of_total(score_1['asr_quality']), "Idea 2": format_score_out_of_total(score_2['asr_quality'])},
        {"Metric": "Dataset Coverage", "Idea 1": format_score_out_of_total(score_1['dataset_coverage']), "Idea 2": format_score_out_of_total(score_2['dataset_coverage'])},
        {"Metric": "Chunk Clarity", "Idea 1": format_score_out_of_total(score_1['chunk_clarity']), "Idea 2": format_score_out_of_total(score_2['chunk_clarity'])},
        {"Metric": "Technical Strength", "Idea 1": format_score_out_of_total(score_1['technical_strength']), "Idea 2": format_score_out_of_total(score_2['technical_strength'])},
        {"Metric": "Successful Chunk Ratio", "Idea 1": format_score_out_of_total(score_1['successful_chunk_ratio']), "Idea 2": format_score_out_of_total(score_2['successful_chunk_ratio'])},
        {"Metric": "Favourability Score", "Idea 1": format_score_out_of_total(score_1['favourability_score']), "Idea 2": format_score_out_of_total(score_2['favourability_score'])},
    ]

    st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

    display_comparison_visuals(score_1, score_2)

    st.subheader("🧠 Dynamic Comparison Feedback")

    for index, item in enumerate(comparison_feedback, start=1):
        st.write(f"{index}. {item}")

    with st.expander("📊 Knowledge-base Evaluation Visuals", expanded=False):
        kb_col_1, kb_col_2 = st.columns(2)

        with kb_col_1:
            display_kb_score_table(
                result_1.get("kb_result"),
                prefix="comparison_idea_1_kb_table_only",
            )

        with kb_col_2:
            display_kb_score_table(
                result_2.get("kb_result"),
                prefix="comparison_idea_2_kb_table_only",
            )

    st.subheader("Idea 1 Debug Match Metric Visuals")
    display_all_match_metric_visuals(
        result_1.get("kb_result"),
        prefix="comparison_idea_1_debug",
    )

    st.subheader("Idea 2 Debug Match Metric Visuals")
    display_all_match_metric_visuals(
        result_2.get("kb_result"),
        prefix="comparison_idea_2_debug",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Idea 1 Refined Transcript")
        st.text_area(
            "Idea 1 complete refined transcript",
            result_1.get("refined_full_transcript", "") or "[No refined transcript]",
            height=350,
            key="idea_1_comparison_refined_transcript",
        )

        idea_1_confidence = result_1.get("audio_pitch_confidence", {})
        st.metric(
            "Idea 1 Audio Pitch Confidence",
            format_score_out_of_total(idea_1_confidence.get('audio_pitch_confidence_rate', 0.0)),
        )

    with c2:
        st.subheader("Idea 2 Refined Transcript")
        st.text_area(
            "Idea 2 complete refined transcript",
            result_2.get("refined_full_transcript", "") or "[No refined transcript]",
            height=350,
            key="idea_2_comparison_refined_transcript",
        )

        idea_2_confidence = result_2.get("audio_pitch_confidence", {})
        st.metric(
            "Idea 2 Audio Pitch Confidence",
            format_score_out_of_total(idea_2_confidence.get('audio_pitch_confidence_rate', 0.0)),
        )

    st.subheader("Detailed Individual Results")

    detail_choice = st.radio(
        "Choose detailed result to view",
        ["Idea 1 Details", "Idea 2 Details"],
        horizontal=True,
        key="comparison_detail_result_radio",
    )

    if detail_choice == "Idea 1 Details":
        display_results(result=result_1, show_debug=globals().get("show_debug", False), prefix="idea_1_details")
    else:
        display_results(result=result_2, show_debug=globals().get("show_debug", False), prefix="idea_2_details")

    comparison_payload = {
        "idea_1_scores": score_1,
        "idea_2_scores": score_2,
        "decision": decision,
        "recommendation": recommendation,
        "comparison_feedback": comparison_feedback,
        "idea_1_result": result_1,
        "idea_2_result": result_2,
    }

    st.download_button(
        "Download comparison JSON",
        data=json.dumps(
            comparison_payload,
            ensure_ascii=False,
            indent=2,
            default=convert_numpy,
        ),
        file_name="two_idea_comparison_result.json",
        mime="application/json",
        use_container_width=True,
        key="download_two_idea_comparison_json",
    )


# ============================================================
# FUTURISTIC CLEAN UI + RESULT AUDIO SUMMARY
# Added in final supervisor-friendly build
# ============================================================

def inject_futuristic_ui():
    """Clean, futuristic, supervisor-friendly style. Does not change the evaluation logic."""
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #07111f;
            --bg-1: #0b1426;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-2: rgba(30, 41, 59, 0.70);
            --line: rgba(148, 163, 184, 0.20);
            --text: #f8fafc;
            --muted: rgba(226, 232, 240, 0.72);
            --blue: #38bdf8;
            --violet: #8b5cf6;
            --green: #22c55e;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(56, 189, 248, 0.18), transparent 28%),
                radial-gradient(circle at 90% 8%, rgba(139, 92, 246, 0.16), transparent 30%),
                radial-gradient(circle at 55% 95%, rgba(34, 197, 94, 0.08), transparent 26%),
                linear-gradient(135deg, #07111f 0%, #0b1426 45%, #111827 100%);
            color: var(--text);
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 1.15rem;
            padding-bottom: 3.2rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.035em;
            font-weight: 900 !important;
            color: #f8fafc !important;
        }

        p, label, span, div {
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .hero-shell {
            position: relative;
            padding: 2.2rem 2.35rem;
            border-radius: 34px;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.92), rgba(124, 58, 237, 0.88)),
                radial-gradient(circle at 20% 12%, rgba(255,255,255,0.32), transparent 26%);
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 30px 80px rgba(37, 99, 235, 0.28);
            overflow: hidden;
            margin-bottom: 1.25rem;
        }

        .hero-shell:after {
            content: "";
            position: absolute;
            inset: -2px;
            background: linear-gradient(120deg, transparent, rgba(255,255,255,0.16), transparent);
            transform: translateX(-70%);
            opacity: 0.45;
        }

        .hero-title {
            position: relative;
            z-index: 1;
            font-size: clamp(2rem, 4.4vw, 3.6rem);
            font-weight: 950;
            line-height: 1.06;
            color: #ffffff;
            margin-bottom: 0.62rem;
        }

        .hero-subtitle {
            position: relative;
            z-index: 1;
            color: rgba(255,255,255,0.90);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 1100px;
        }

        .pill-row {
            position: relative;
            z-index: 1;
            display: flex;
            flex-wrap: wrap;
            gap: 0.62rem;
            margin-top: 1.15rem;
        }

        .pill {
            padding: 0.48rem 0.82rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.24);
            color: white;
            font-weight: 850;
            font-size: 0.88rem;
        }

        .model-panel {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1.05rem 0 1.35rem;
        }

        .model-card {
            padding: 1.05rem 1.1rem;
            border-radius: 24px;
            background: rgba(15,23,42,0.78);
            border: 1px solid rgba(148,163,184,0.20);
            box-shadow: 0 14px 34px rgba(0,0,0,0.22);
        }

        .model-kicker {
            color: #60a5fa;
            font-size: 0.75rem;
            font-weight: 950;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 0.32rem;
        }

        .model-title {
            color: #ffffff;
            font-size: 1.04rem;
            font-weight: 900;
            margin-bottom: 0.24rem;
        }

        .model-text {
            color: rgba(226,232,240,0.74);
            font-size: 0.9rem;
            line-height: 1.56;
        }

        .result-listen-card {
            padding: 1rem 1.1rem;
            border-radius: 22px;
            background: rgba(14, 165, 233, 0.11);
            border: 1px solid rgba(56, 189, 248, 0.24);
            margin: 0.95rem 0 1.15rem;
        }

        .result-listen-title {
            font-size: 1.05rem;
            font-weight: 900;
            color: #e0f2fe;
            margin-bottom: 0.25rem;
        }

        .result-listen-text {
            color: rgba(224,242,254,0.82);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
            border: 1px solid rgba(148,163,184,0.20);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 12px 30px rgba(0,0,0,0.20);
        }

        div[data-testid="stMetricLabel"] {
            color: rgba(226,232,240,0.86) !important;
            font-weight: 850;
        }

        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 950;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 16px !important;
            font-weight: 850 !important;
            border: 1px solid rgba(255,255,255,0.16) !important;
            box-shadow: 0 14px 30px rgba(37,99,235,0.24);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
            color: white !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
            border-right: 1px solid rgba(148,163,184,0.16);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.55rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            padding: 0.65rem 1.05rem;
            font-weight: 850;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
            color: white !important;
        }

        .stExpander {
            border-radius: 18px !important;
            background: rgba(255,255,255,0.045) !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        }

        textarea {
            border-radius: 16px !important;
            line-height: 1.85 !important;
            font-size: 16px !important;
        }

        .new-user-note {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(34,197,94,0.10);
            border: 1px solid rgba(74,222,128,0.20);
            color: rgba(220,252,231,0.94);
            margin: 0.85rem 0 1.1rem;
            line-height: 1.62;
        }

        @media (max-width: 1050px) {
            .model-panel { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
            .model-panel { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_futuristic_hero():
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-title">🎙️ Tamil Idea Pitch Evaluation System</div>
            <div class="hero-subtitle">
                Upload a Tamil audio or video pitch and get a structured evaluation with transcript analysis,
                ML confidence prediction, knowledge-base scoring, budget feasibility, and two-idea comparison.
                The audio narration is available after results are generated, so it explains the actual output — not the process.
            </div>
            <div class="pill-row">
                <span class="pill">Audio + Video Input</span>
                <span class="pill">Tamil STT / ASR</span>
                <span class="pill">ML Confidence Model</span>
                <span class="pill">Knowledge-base Scoring</span>
                <span class="pill">Budget Feasibility</span>
                <span class="pill">Result Voice Summary</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="model-panel">
            <div class="model-card">
                <div class="model-kicker">Input Layer</div>
                <div class="model-title">Audio / Video Pitch</div>
                <div class="model-text">Upload Tamil audio or video. Video audio is extracted automatically using FFmpeg support.</div>
            </div>
            <div class="model-card">
                <div class="model-kicker">Speech Layer</div>
                <div class="model-title">Tamil STT + Chunks</div>
                <div class="model-text">The media is split into chunks, transcribed, refined, and prepared for scoring.</div>
            </div>
            <div class="model-card">
                <div class="model-kicker">ML Layer</div>
                <div class="model-title">Confidence Model</div>
                <div class="model-text">The trained model predicts confidence signals from transcript features.</div>
            </div>
            <div class="model-card">
                <div class="model-kicker">Evaluation Layer</div>
                <div class="model-title">KB + Budget + Feedback</div>
                <div class="model-text">Scores, visuals, budget estimate, recommendation, and result narration are generated.</div>
            </div>
        </div>
        <div class="new-user-note">
            <b>For new users:</b> choose an evaluation mode from the sidebar, upload Tamil audio/video, then run evaluation. After execution, use the result audio button to hear the generated result summary.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_panel_sidebar(model_path, kb_path, text_dataset_path):
    with st.sidebar:
        st.markdown("## ⚙️ Model Panel")
        st.caption("Project paths and evaluation controls")
        st.markdown("---")
        st.markdown("**Active Modules**")
        st.markdown("- Tamil STT / ASR")
        st.markdown("- ML confidence model")
        st.markdown("- Tamil knowledge base")
        st.markdown("- Budget feasibility")
        st.markdown("- Single + comparison mode")
        st.markdown("- Result voice narration")
        st.markdown("---")
        st.caption("Torch/Whisper toggle is intentionally removed for deployment-safe testing.")


def result_audio_button(label: str, narration: str, key: str):
    """Browser-based text-to-speech. Click again stops current speech."""
    safe_label = json.dumps(str(label), ensure_ascii=False)
    narration_json = json.dumps(str(narration), ensure_ascii=False)
    safe_key = ''.join(ch if ch.isalnum() else '_' for ch in str(key))

    components.html(
        f"""
        <button id="btn_{safe_key}" style="
            width: 100%;
            border: 1px solid rgba(96,165,250,0.48);
            background: linear-gradient(135deg, rgba(37,99,235,0.96), rgba(124,58,237,0.94));
            color: white;
            border-radius: 16px;
            padding: 12px 16px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 850;
            box-shadow: 0 12px 28px rgba(37,99,235,0.26);
        ">🔊 " + {safe_label} + "</button>
        <script>
        const button_{safe_key} = document.getElementById("btn_{safe_key}");
        let isSpeaking_{safe_key} = false;
        const originalLabel_{safe_key} = "🔊 " + {safe_label};
        button_{safe_key}.onclick = function() {{
            const text = {narration_json};
            if (window.parent.speechSynthesis.speaking || isSpeaking_{safe_key}) {{
                window.parent.speechSynthesis.cancel();
                isSpeaking_{safe_key} = false;
                button_{safe_key}.innerText = originalLabel_{safe_key};
                return;
            }}
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.92;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            utterance.onend = function() {{
                isSpeaking_{safe_key} = false;
                button_{safe_key}.innerText = originalLabel_{safe_key};
            }};
            utterance.onerror = function() {{
                isSpeaking_{safe_key} = false;
                button_{safe_key}.innerText = originalLabel_{safe_key};
            }};
            window.parent.speechSynthesis.cancel();
            window.parent.speechSynthesis.speak(utterance);
            isSpeaking_{safe_key} = true;
            button_{safe_key}.innerText = "⏹ Stop result audio";
        }};
        </script>
        """,
        height=52,
    )


def _safe_score_10(value):
    return round(normalize_score(value) * 10.0, 2)


def build_single_result_narration(result, title="this idea"):
    chunks = result.get("chunks", [])
    refined_text = result.get("refined_full_transcript", "") or ""
    asr_quality = float(result.get("asr_quality", 0.0))
    audio_conf = result.get("audio_pitch_confidence", {}) or {}
    audio_rate = float(audio_conf.get("audio_pitch_confidence_rate", 0.0))
    audio_label = audio_conf.get("audio_pitch_confidence_label", "not available")
    kb_result = result.get("kb_result", {}) or {}
    kb_scores = kb_result.get("scores", {}) or {}
    final_score = _safe_score_10(kb_result.get("final_score", 0.0))
    technical = _safe_score_10(kb_scores.get("technical_correctness", 0.0))
    real_life = _safe_score_10(kb_scores.get("real_life_probability", 0.0))
    theory = _safe_score_10(kb_scores.get("theoretical_correctness", 0.0))
    budget = result.get("budget_feasibility", {}) or {}
    budget_score = budget.get("budget_feasibility_out_of_10", None)
    budget_label = budget.get("budget_feasibility_label", "not available")
    budget_estimate = (budget.get("estimation", {}) or {}).get("total_estimated_budget", "not available")

    word_count = len(refined_text.split()) if refined_text else 0
    chunk_count = len(chunks)

    parts = [
        f"Result summary for {title}.",
        f"The system processed {chunk_count} chunks and detected around {word_count} transcript words.",
        f"ASR quality is {_safe_score_10(asr_quality)} out of 10.",
        f"Audio pitch confidence is {_safe_score_10(audio_rate)} out of 10, classified as {audio_label}.",
        f"Knowledge base final score is {final_score} out of 10.",
        f"Technical correctness is {technical} out of 10, real life probability is {real_life} out of 10, and theoretical correctness is {theory} out of 10.",
    ]

    if budget_score is not None:
        parts.append(f"Budget feasibility is {budget_score} out of 10, classified as {budget_label}. Estimated budget range is {budget_estimate}.")

    if final_score >= 7:
        parts.append("Overall, this pitch is strong and ready for presentation with minor improvements.")
    elif final_score >= 4.5:
        parts.append("Overall, this pitch is understandable, but it needs clearer technical details, real world use case, and stronger explanation.")
    else:
        parts.append("Overall, this pitch needs improvement. Add clearer problem statement, implementation method, cost details, target users, and practical benefits.")

    return " ".join(parts)


def render_single_result_audio(result, prefix="single"):
    narration = build_single_result_narration(result, title="this idea")
    st.markdown(
        """
        <div class="result-listen-card">
            <div class="result-listen-title">🔊 Result Audio Summary</div>
            <div class="result-listen-text">This button reads the generated result after execution. Click again to stop.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    result_audio_button("Recite this result", narration, f"{prefix}_result_audio")


def build_comparison_narration(result_1, result_2):
    score_1 = calculate_favourability_score(result_1)
    score_2 = calculate_favourability_score(result_2)
    decision = decide_better_idea(score_1, score_2)

    fav_1 = format_score_out_of_total(score_1.get("favourability_score", 0.0))
    fav_2 = format_score_out_of_total(score_2.get("favourability_score", 0.0))

    return (
        "Comparison result summary. "
        f"Idea 1 favourability score is {fav_1}. "
        f"Idea 2 favourability score is {fav_2}. "
        f"The system decision is: {decision.get('winner', 'not available')}. "
        f"Reason: {decision.get('reason', 'No reason available')}. "
        "Use the comparison score table and dynamic feedback to identify which idea has stronger technical correctness, real life probability, audio confidence, and knowledge base alignment."
    )


def render_comparison_result_audio(result_1, result_2):
    narration = build_comparison_narration(result_1, result_2)
    st.markdown(
        """
        <div class="result-listen-card">
            <div class="result-listen-title">🔊 Comparison Audio Summary</div>
            <div class="result-listen-text">This button reads the comparison decision after both ideas are evaluated. Click again to stop.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    result_audio_button("Recite comparison result", narration, "comparison_result_audio")


def validate_common_paths(model_path, kb_path, text_dataset_path):
    if not os.path.exists(model_path):
        st.error(f"Confidence model not found: {model_path}")
        return False

    if not os.path.exists(kb_path):
        st.error(f"Knowledge base not found: {kb_path}")
        return False

    if not os.path.exists(text_dataset_path):
        st.error(f"Tamil text dataset not found: {text_dataset_path}")
        return False

    return True


if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None

if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

if "last_uploaded_audio_name" not in st.session_state:
    st.session_state.last_uploaded_audio_name = None

if "last_idea_1_name" not in st.session_state:
    st.session_state.last_idea_1_name = None

if "last_idea_2_name" not in st.session_state:
    st.session_state.last_idea_2_name = None

st.set_page_config(
    page_title="Tamil Idea Pitch Evaluator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_futuristic_ui()
render_futuristic_hero()

with st.sidebar:
    st.markdown("## 🎛️ Control Center")
    st.caption("Choose mode, verify model paths, and run evaluation.")

    if st.button("Clear Previous Results / Cache", use_container_width=True):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

    app_mode = st.radio(
        "Evaluation Mode",
        ["Single Idea Evaluation", "Compare Two Ideas"],
        help="Use single mode for one pitch, or comparison mode to choose the stronger idea.",
    )

    show_debug = st.toggle(
        "Show technical debug tables",
        value=False,
        help="Enable only for debugging transcript chunks and STT details.",
    )

    st.markdown("---")
    model_path = st.text_input("Confidence model path", MODEL_PATH)
    kb_path = st.text_input("Knowledge base path", KB_PATH)
    text_dataset_path = st.text_input("Tamil text dataset path", TEXT_DATASET_PATH)

render_model_panel_sidebar(model_path, kb_path, text_dataset_path)

st.markdown("---")

if app_mode == "Single Idea Evaluation":
    left, right = st.columns([1.15, 0.85])

    with left:
        st.header("🧠 Single Tamil Idea Evaluation")
        st.caption("Upload one Tamil audio/video pitch. The system will transcribe, evaluate, score, and generate result narration after execution.")

        uploaded_audio = st.file_uploader(
            "Upload Tamil audio/video idea pitch",
            type=UPLOAD_TYPES,
            accept_multiple_files=False,
            key="single_uploaded_audio",
            help="Supported: WAV, MP3, M4A, FLAC, OGG, MP4, MOV, MKV, AVI, WEBM",
        )

        if uploaded_audio is not None:
            preview_uploaded_media(uploaded_audio, "Uploaded Tamil audio/video")

            if st.session_state.last_uploaded_audio_name != uploaded_audio.name:
                st.session_state.evaluation_result = None
                st.session_state.last_uploaded_audio_name = uploaded_audio.name

        run_button = st.button(
            "🚀 Run Evaluation",
            type="primary",
            use_container_width=True,
            key="single_run_evaluation_button",
        )

    with right:
        st.markdown("### 📌 What you will get")
        st.info(
            "After execution, the dashboard shows transcript, ASR quality, chunk statistics, ML confidence, knowledge-base score, budget feasibility, feedback, and a result audio summary."
        )
        st.markdown("### ✅ Best pitch tips")
        st.write("- State the problem clearly")
        st.write("- Mention target users")
        st.write("- Explain technology stack")
        st.write("- Include budget/deployment details")
        st.write("- Speak clearly with less background noise")

    if run_button:
        if uploaded_audio is None:
            st.error("Please upload an audio or video file first.")

        elif validate_common_paths(model_path, kb_path, text_dataset_path):
            temp_audio_path = None

            try:
                temp_audio_path = save_uploaded_audio(uploaded_audio)

                with st.spinner("Extracting audio if needed, transcribing media, and evaluating the idea..."):
                    result = run_pipeline(
                        temp_audio_path,
                        model_path,
                        kb_path,
                        text_dataset_path,
                        prefer_torch=False,
                    )

                st.session_state.evaluation_result = result
                st.session_state.last_uploaded_audio_name = uploaded_audio.name
                st.success("Evaluation completed. Result audio summary is now available below.")

            except Exception as e:
                st.error(str(e))
                st.code(traceback.format_exc(), language="python")

            finally:
                if temp_audio_path and os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

    if st.session_state.evaluation_result is not None:
        render_single_result_audio(st.session_state.evaluation_result, prefix="single")
        display_results(
            result=st.session_state.evaluation_result,
            show_debug=show_debug,
            prefix="single",
        )


elif app_mode == "Compare Two Ideas":
    st.header("⚖️ Compare Two Tamil Idea Pitches")
    st.caption("Upload two Tamil audio/video pitches. The system evaluates both with the same pipeline and recommends the stronger idea.")

    upload_col_1, upload_col_2 = st.columns(2)

    with upload_col_1:
        idea_1_audio = st.file_uploader(
            "Upload Idea 1 Tamil audio/video pitch",
            type=UPLOAD_TYPES,
            accept_multiple_files=False,
            key="idea_1_audio_uploader",
        )

        if idea_1_audio is not None:
            preview_uploaded_media(idea_1_audio, "Idea 1 Audio/Video")

    with upload_col_2:
        idea_2_audio = st.file_uploader(
            "Upload Idea 2 Tamil audio/video pitch",
            type=UPLOAD_TYPES,
            accept_multiple_files=False,
            key="idea_2_audio_uploader",
        )

        if idea_2_audio is not None:
            preview_uploaded_media(idea_2_audio, "Idea 2 Audio/Video")

    if (
        idea_1_audio is not None
        and idea_2_audio is not None
        and (
            st.session_state.last_idea_1_name != idea_1_audio.name
            or st.session_state.last_idea_2_name != idea_2_audio.name
        )
    ):
        st.session_state.comparison_result = None
        st.session_state.last_idea_1_name = idea_1_audio.name
        st.session_state.last_idea_2_name = idea_2_audio.name

    compare_button = st.button(
        "⚡ Compare Both Ideas",
        type="primary",
        use_container_width=True,
        key="compare_both_ideas_button",
    )

    if compare_button:
        if idea_1_audio is None or idea_2_audio is None:
            st.error("Please upload both Idea 1 and Idea 2 audio/video files.")

        elif validate_common_paths(model_path, kb_path, text_dataset_path):
            idea_1_temp = None
            idea_2_temp = None

            try:
                idea_1_temp = save_uploaded_audio(idea_1_audio)
                idea_2_temp = save_uploaded_audio(idea_2_audio)

                with st.spinner("Analyzing Idea 1..."):
                    result_1 = run_pipeline(
                        idea_1_temp,
                        model_path,
                        kb_path,
                        text_dataset_path,
                        prefer_torch=False,
                    )

                with st.spinner("Analyzing Idea 2..."):
                    result_2 = run_pipeline(
                        idea_2_temp,
                        model_path,
                        kb_path,
                        text_dataset_path,
                        prefer_torch=False,
                    )

                st.session_state.comparison_result = {
                    "idea_1": result_1,
                    "idea_2": result_2,
                }
                st.success("Comparison completed. Comparison audio summary is now available below.")

            except Exception as e:
                st.error(str(e))
                st.code(traceback.format_exc(), language="python")

            finally:
                if idea_1_temp and os.path.exists(idea_1_temp):
                    os.remove(idea_1_temp)

                if idea_2_temp and os.path.exists(idea_2_temp):
                    os.remove(idea_2_temp)

    if st.session_state.comparison_result is not None:
        render_comparison_result_audio(
            st.session_state.comparison_result["idea_1"],
            st.session_state.comparison_result["idea_2"],
        )
        display_comparison_results(
            result_1=st.session_state.comparison_result["idea_1"],
            result_2=st.session_state.comparison_result["idea_2"],
            show_debug=show_debug,
        )


with st.expander("How this website works", expanded=False):
    st.markdown(
        """
        **Single Idea Evaluation**
        1. Upload one Tamil audio or video pitch.
        2. The system extracts audio when needed and splits the pitch into chunks.
        3. Each chunk is transcribed, refined, and analyzed.
        4. The website shows transcript, chunk statistics, audio confidence, knowledge-base score, budget feasibility, and feedback.
        5. After execution, click the result audio button to hear a spoken summary of the actual output.

        **Compare Two Ideas**
        1. Upload two Tamil audio/video pitches.
        2. Each idea is evaluated using the same pipeline.
        3. The comparison uses technical correctness, real-life probability, theoretical correctness, knowledge-base score, audio confidence, ASR quality, and dataset coverage.
        4. The system recommends the more favourable idea and can recite the comparison result after execution.
        """
    )
