
import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from cocoa_detector import detect_pods_with_edge_impulse
from PIL import Image, ImageOps


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="CocoaTrack",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>

/* ===== HOME HERO OVERRIDES ===== */
.ct-home-title{
    font-size:4.4rem !important;
    line-height:1.04 !important;
    max-width:100% !important;
}
.ct-home-description{
    max-width:100% !important;
    font-size:1.18rem !important;
    line-height:1.75 !important;
}

        :root {
            --page: #f5efe7;
            --surface: rgba(255,255,255,0.92);
            --surface-soft: #fbf7f2;
            --ink: #2a211c;
            --muted: #74685f;
            --line: #e4d9cf;
            --cocoa: #513525;
            --cocoa-2: #7a523a;
            --terracotta: #c97a57;
            --gold: #d6a158;
            --leaf: #71876a;
            --leaf-soft: #e8efe4;
            --shadow: 0 18px 42px rgba(83, 54, 35, 0.10);
        }

        html, body, [class*="css"] {
            font-family:
                "Avenir Next",
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(214,161,88,0.18), transparent 24%),
                radial-gradient(circle at 95% 7%, rgba(113,135,106,0.16), transparent 28%),
                linear-gradient(180deg, #fbf7f2 0%, var(--page) 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #f6eee6 0%, #efe3d8 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color: var(--ink);
        }

        [data-testid="stSidebar"] label {
            font-weight: 700;
            color: var(--ink) !important;
        }

        [data-testid="stSidebar"] input {
            background: white !important;
            color: var(--ink) !important;
            border: 1px solid #cdbfb3 !important;
            border-radius: 12px !important;
        }

        .app-header {
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            background:
                linear-gradient(135deg, #5a3827 0%, #7d5842 55%, #8c6b4f 100%);
            border-radius: 26px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: var(--shadow);
        }

        .app-header::after {
            content: "✦";
            position: absolute;
            right: 1.4rem;
            top: -0.2rem;
            font-size: 6.5rem;
            color: rgba(255,255,255,0.07);
            transform: rotate(10deg);
        }

        .brand-group {
            display: flex;
            align-items: center;
            gap: 0.95rem;
            z-index: 1;
        }

        .brand-mark {
            width: 54px;
            height: 54px;
            border-radius: 18px;
            display: grid;
            place-items: center;
            background:
                linear-gradient(135deg, var(--gold), #b7673f);
            color: white;
            font-size: 1.45rem;
            box-shadow: 0 12px 24px rgba(40,20,10,0.20);
            animation: cocoaFloat 3.2s ease-in-out infinite;
        }

        @keyframes cocoaFloat {
            0%,100% { transform: translateY(0); }
            50% { transform: translateY(-4px); }
        }

        .brand-title {
            color: white;
            font-size: 1.6rem;
            font-weight: 850;
            letter-spacing: -0.03em;
        }

        .brand-subtitle {
            color: #f2e7df;
            font-size: 0.9rem;
            margin-top: 0.15rem;
        }

        .header-badge {
            z-index: 1;
            background: rgba(255,255,255,0.15);
            color: white;
            border: 1px solid rgba(255,255,255,0.20);
            border-radius: 999px;
            padding: 0.5rem 0.8rem;
            font-size: 0.78rem;
            font-weight: 800;
            backdrop-filter: blur(8px);
        }

        div[role="radiogroup"] {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-bottom: 0.9rem;
        }

        div[role="radiogroup"] label {
            background: rgba(255,255,255,0.90);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.45rem 0.7rem;
            box-shadow: 0 6px 16px rgba(83,54,35,0.05);
            transition: 0.18s ease;
        }

        div[role="radiogroup"] label:hover {
            transform: translateY(-2px);
            border-color: #cdb8a7;
            background: #fffdf9;
        }

        .home-panel {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, #4f3325 0%, #7d5238 62%, #a46a48 100%);
            border-radius: 28px;
            padding: 2.1rem;
            margin-bottom: 1.15rem;
            box-shadow: var(--shadow);
        }

        .home-panel::before {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -70px;
            bottom: -100px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.14), transparent 70%);
        }

        .home-panel h1 {
            color: white;
            font-size: clamp(2.5rem, 5vw, 4rem);
            line-height: 0.98;
            letter-spacing: -0.045em;
            margin: 0 0 0.7rem;
            max-width: 850px;
        }

        .home-panel p {
            color: #f3e7de;
            max-width: 760px;
            font-size: 1.02rem;
            line-height: 1.65;
            margin: 0;
        }

        .feature-card,
        .class-card {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,246,241,0.95));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.2rem;
            box-shadow: 0 12px 30px rgba(83,54,35,0.08);
            transition: 0.18s ease;
        }

        .feature-card:hover,
        .class-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 38px rgba(83,54,35,0.12);
        }

        .feature-card {
            min-height: 150px;
        }

        .feature-icon {
            width: 44px;
            height: 44px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: var(--leaf-soft);
            font-size: 1.45rem;
            margin-bottom: 0.7rem;
        }

        .feature-title {
            color: var(--ink);
            font-weight: 850;
            font-size: 1.02rem;
            margin-bottom: 0.3rem;
        }

        .feature-text,
        .class-card span {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .class-card {
            text-align: center;
            border-top: 5px solid var(--gold);
        }

        .class-card strong {
            color: var(--ink);
            font-size: 1rem;
        }

        .analysis-path {
            background:
                linear-gradient(90deg, rgba(214,161,88,0.11), rgba(113,135,106,0.11));
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: var(--muted);
            text-align: center;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .stage-box {
            display: flex;
            align-items: flex-start;
            gap: 0.95rem;
            background: rgba(255,255,255,0.94);
            border: 1px solid var(--line);
            border-left: 6px solid var(--terracotta);
            border-radius: 20px;
            padding: 1.05rem 1.15rem;
            margin-top: 1.5rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 12px 28px rgba(83,54,35,0.07);
        }

        .stage-number {
            width: 42px;
            height: 42px;
            min-width: 42px;
            border-radius: 13px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, var(--cocoa), var(--cocoa-2));
            color: white;
            font-size: 0.9rem;
            font-weight: 850;
        }

        .stage-copy h3 {
            margin: 0;
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 850;
        }

        .stage-copy p {
            margin: 0.2rem 0 0;
            color: var(--muted);
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            background:
                linear-gradient(180deg, #ffffff, #faf7f3);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem;
            box-shadow: 0 10px 24px rgba(83,54,35,0.06);
        }

        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 850;
        }

        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.96);
            border: 1.5px dashed #cdb6a4;
            border-radius: 18px;
            padding: 0.5rem;
            box-shadow: 0 10px 24px rgba(83,54,35,0.05);
        }

        [data-testid="stImage"] img {
            border-radius: 18px;
            border: 1px solid var(--line);
            box-shadow: 0 14px 32px rgba(83,54,35,0.08);
        }

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--line);
            box-shadow: 0 10px 24px rgba(83,54,35,0.05);
        }

        .note-box,
        .explain-card {
            background:
                linear-gradient(135deg, #fffaf4, #f7f1e9);
            border: 1px solid var(--line);
            border-left: 5px solid var(--gold);
            border-radius: 16px;
            padding: 1rem;
            color: #574c45;
            line-height: 1.55;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            min-height: 3.45rem;
            border-radius: 16px !important;
            padding: 0.8rem 1.2rem !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.01em;
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid #cdbfb3 !important;
            box-shadow: 0 10px 22px rgba(83,54,35,0.08) !important;
            transition: 0.18s ease !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            background: #fffaf5 !important;
            border-color: #bda896 !important;
            box-shadow: 0 16px 28px rgba(83,54,35,0.12) !important;
        }

        .stButton > button[kind="primary"] {
            background:
                linear-gradient(135deg, var(--terracotta), var(--cocoa-2)) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 14px 28px rgba(125,82,56,0.22) !important;
        }

        .stButton > button[kind="primary"]:hover {
            background:
                linear-gradient(135deg, #b76d4f, #654431) !important;
            transform: translateY(-2px);
        }

        details {
            background: rgba(255,255,255,0.92);
            border: 1px solid var(--line);
            border-radius: 16px;
        }

        .footer-note {
            text-align: center;
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 1.6rem;
        }

        @media (max-width: 820px) {
            .app-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# FINAL UI POLISH
# Visual only — no pipeline logic is changed.
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL PAGE
       ------------------------------------------------------- */

    .block-container {
        max-width: 1240px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
    }

    p {
        line-height: 1.55;
    }

    /* -------------------------------------------------------
       HEADER
       ------------------------------------------------------- */

    .topbar {
        padding: 1rem 1.25rem !important;
        border-radius: 18px !important;
        box-shadow:
            0 1px 2px rgba(45, 32, 24, 0.03),
            0 10px 30px rgba(45, 32, 24, 0.05) !important;
    }

    .brand-mark {
        width: 44px !important;
        height: 44px !important;
        border-radius: 13px !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12);
    }

    .brand-title {
        font-size: 1.3rem !important;
        letter-spacing: -0.03em;
    }

    .brand-subtitle {
        font-size: 0.82rem !important;
    }

    .status-pill {
        background: #edf3ea !important;
        border: 1px solid #d9e5d5;
        padding: 0.42rem 0.72rem !important;
    }

    /* -------------------------------------------------------
       HERO
       ------------------------------------------------------- */

    .intro-card {
        position: relative;
        overflow: hidden;
        padding: 2rem 2rem !important;
        border-radius: 22px !important;
        background:
            radial-gradient(
                circle at 88% 15%,
                rgba(255,255,255,0.12),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #4b2e20 0%,
                #654432 58%,
                #76543f 100%
            ) !important;
        box-shadow:
            0 16px 38px rgba(62, 40, 27, 0.16) !important;
    }

    .intro-title {
        font-size: clamp(1.8rem, 3vw, 2.45rem) !important;
        max-width: 700px;
    }

    .intro-text {
        max-width: 680px !important;
        line-height: 1.6 !important;
        opacity: 0.95;
    }

    /* -------------------------------------------------------
       STEPPER
       ------------------------------------------------------- */

    .stepper {
        gap: 0.5rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1.6rem !important;
    }

    .step {
        padding: 0.72rem 0.65rem !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.72) !important;
        box-shadow: 0 2px 8px rgba(40,30,24,0.025);
    }

    /* -------------------------------------------------------
       SECTION HEADINGS
       ------------------------------------------------------- */

    .stage-box {
        background: rgba(255,255,255,0.64);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.85rem 1rem;
        margin-top: 1.7rem !important;
        margin-bottom: 1rem !important;
    }

    .stage-index {
        width: 36px !important;
        height: 36px !important;
        border-radius: 11px !important;
        box-shadow: 0 4px 12px rgba(91,58,41,0.14);
    }

    .stage-copy h3 {
        font-size: 1.08rem !important;
    }

    .stage-copy p {
        max-width: 850px;
        line-height: 1.45;
    }

    /* -------------------------------------------------------
       METRICS
       ------------------------------------------------------- */

    div[data-testid="stMetric"] {
        min-height: 104px;
        padding: 1rem 1.05rem !important;
        border-radius: 16px !important;
        border: 1px solid #e4ddd4 !important;
        background:
            linear-gradient(
                180deg,
                #ffffff 0%,
                #fcfaf7 100%
            ) !important;
        box-shadow:
            0 2px 6px rgba(50,38,29,0.025),
            0 10px 24px rgba(50,38,29,0.035) !important;
    }

    div[data-testid="stMetric"] label {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.035em;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        line-height: 1.1 !important;
    }

    /* -------------------------------------------------------
       IMAGES
       ------------------------------------------------------- */

    [data-testid="stImage"] {
        margin-bottom: 0.35rem;
    }

    [data-testid="stImage"] img {
        border-radius: 15px !important;
        border: 1px solid #ddd5cb !important;
        box-shadow:
            0 3px 8px rgba(48,36,27,0.04),
            0 12px 28px rgba(48,36,27,0.06);
    }

    [data-testid="stImage"] div {
        color: var(--muted);
    }

    /* -------------------------------------------------------
       BUTTONS
       ------------------------------------------------------- */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px !important;
        min-height: 2.75rem !important;
        transition:
            transform 0.12s ease,
            box-shadow 0.12s ease,
            border-color 0.12s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        border-color: #c9b9aa !important;
        box-shadow: 0 6px 16px rgba(57,42,31,0.07) !important;
    }

    .stButton > button[kind="primary"] {
        background:
            linear-gradient(
                135deg,
                #563626,
                #684735
            ) !important;
    }

    /* -------------------------------------------------------
       RADIO / SELECT / INPUT
       ------------------------------------------------------- */

    div[data-testid="stRadio"] > div {
        gap: 0.4rem;
    }

    div[data-testid="stRadio"] label {
        border-radius: 999px;
    }

    [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        border-radius: 10px !important;
    }

    /* -------------------------------------------------------
       UPLOAD
       ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        border-radius: 16px !important;
        background: rgba(255,255,255,0.7) !important;
        border: 1px dashed #cdbfae !important;
    }

    [data-testid="stFileUploader"]:hover {
        background: #ffffff !important;
        border-color: #ae927c !important;
    }

    /* -------------------------------------------------------
       DATAFRAMES
       ------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border-radius: 15px !important;
        border: 1px solid #e1d9cf !important;
        box-shadow: 0 5px 18px rgba(48,36,27,0.035);
        background: white;
    }

    /* -------------------------------------------------------
       EXPANDERS
       ------------------------------------------------------- */

    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.78) !important;
        overflow: hidden;
        margin: 0.55rem 0;
    }

    [data-testid="stExpander"] summary {
        font-weight: 700;
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }

    /* -------------------------------------------------------
       INFO / WARNING / SUCCESS
       ------------------------------------------------------- */

    [data-testid="stAlert"] {
        border-radius: 13px !important;
        border-width: 1px !important;
    }

    .note-box {
        border-radius: 13px !important;
        line-height: 1.55 !important;
        box-shadow: 0 3px 12px rgba(50,38,29,0.025);
    }

    /* -------------------------------------------------------
       DIVIDERS
       ------------------------------------------------------- */

    hr {
        border: none !important;
        border-top: 1px solid #e5ded6 !important;
        margin: 1.5rem 0 !important;
    }

    /* -------------------------------------------------------
       SIDEBAR
       ------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #2e2621 0%,
                #382d27 100%
            ) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.07);
        border-color: rgba(255,255,255,0.10);
        color: white;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.065) !important;
        border-color: rgba(255,255,255,0.10) !important;
        backdrop-filter: blur(6px);
    }

    /* -------------------------------------------------------
       CAPTIONS
       ------------------------------------------------------- */

    [data-testid="stCaptionContainer"] {
        color: #837970 !important;
        font-size: 0.8rem !important;
        line-height: 1.45 !important;
    }

    /* -------------------------------------------------------
       SPINNER
       ------------------------------------------------------- */

    [data-testid="stSpinner"] {
        color: var(--cocoa);
    }

    /* -------------------------------------------------------
       MOBILE / SMALL WINDOWS
       ------------------------------------------------------- */

    @media (max-width: 900px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .intro-card {
            padding: 1.5rem !important;
        }

        .intro-title {
            font-size: 1.75rem !important;
        }

        div[data-testid="stMetric"] {
            min-height: auto;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CLEAN MINIMAL UI
# Visual only — no pipeline logic changed.
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1180px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
    }

    .stApp {
        background: #f7f5f1 !important;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.015em;
    }

    /* Header */

    .topbar {
        background: #ffffff !important;
        border: 1px solid #e5dfd7 !important;
        border-radius: 14px !important;
        padding: 0.95rem 1.1rem !important;
        box-shadow: none !important;
        margin-bottom: 0.9rem !important;
    }

    .brand-mark {
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    .brand-title {
        font-size: 1.25rem !important;
    }

    .brand-subtitle {
        font-size: 0.82rem !important;
    }

    .status-pill {
        background: #eef2eb !important;
        border: 1px solid #dde5d8 !important;
        color: #51604d !important;
        box-shadow: none !important;
    }

    /* Intro */

    .intro-card {
        background: #5b3a29 !important;
        border-radius: 16px !important;
        padding: 1.6rem 1.7rem !important;
        box-shadow: none !important;
        margin-bottom: 0.9rem !important;
    }

    .intro-title {
        font-size: 1.9rem !important;
    }

    .intro-text {
        max-width: 720px !important;
        line-height: 1.55 !important;
    }

    /* Stepper */

    .stepper {
        gap: 0.45rem !important;
        margin: 0.8rem 0 1.2rem !important;
    }

    .step {
        background: #ffffff !important;
        border: 1px solid #e5dfd7 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        padding: 0.65rem !important;
    }

    /* Section headings */

    .stage-box {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid #e4ded6 !important;
        border-radius: 0 !important;
        padding: 0 0 0.75rem 0 !important;
        margin-top: 1.6rem !important;
        margin-bottom: 1rem !important;
    }

    .stage-index {
        width: 32px !important;
        height: 32px !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    .stage-copy h3 {
        font-size: 1.05rem !important;
    }

    .stage-copy p {
        font-size: 0.88rem !important;
    }

    /* Metrics */

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5dfd7 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        padding: 0.85rem 0.95rem !important;
        min-height: 92px;
    }

    div[data-testid="stMetric"] label {
        color: #7a736c !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    /* Images */

    [data-testid="stImage"] img {
        border-radius: 10px !important;
        border: 1px solid #ded8d0 !important;
        box-shadow: none !important;
    }

    /* Buttons */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px !important;
        min-height: 2.55rem !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: none !important;
        box-shadow: none !important;
    }

    .stButton > button[kind="primary"] {
        background: #5b3a29 !important;
        color: white !important;
        border: 1px solid #5b3a29 !important;
    }

    /* Inputs */

    [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        border-radius: 8px !important;
    }

    /* Upload */

    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px dashed #cfc6bc !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    /* Tables */

    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid #e2dcd4 !important;
        box-shadow: none !important;
    }

    /* Expanders */

    [data-testid="stExpander"] {
        border: 1px solid #e3ddd5 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        box-shadow: none !important;
    }

    /* Alerts */

    [data-testid="stAlert"] {
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    .note-box {
        background: #faf8f5 !important;
        border: 1px solid #e3ddd5 !important;
        border-left: 3px solid #b98546 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background: #312a25 !important;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    /* Captions */

    [data-testid="stCaptionContainer"] {
        color: #7d756e !important;
        font-size: 0.79rem !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid #e4ded6 !important;
        margin: 1.3rem 0 !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .intro-card {
            padding: 1.3rem !important;
        }

        .intro-title {
            font-size: 1.6rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>

    .maturity-grid-title {
        margin-top: 0.25rem;
        margin-bottom: 0.8rem;
    }

    .maturity-card {
        background: #ffffff;
        border: 1px solid #e3ddd5;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        min-height: 92px;
    }

    .maturity-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.4rem;
    }

    .maturity-pod-number {
        font-size: 0.82rem;
        color: #7b746d;
        font-weight: 600;
    }

    .maturity-status {
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid;
    }

    .maturity-ripe {
        background: #eef4eb;
        color: #4d6749;
        border-color: #d8e4d3;
    }

    .maturity-half {
        background: #fbf3e4;
        color: #8a682d;
        border-color: #eadbb9;
    }

    .maturity-unripe {
        background: #f1f5ee;
        color: #567150;
        border-color: #dbe5d7;
    }

    .maturity-uncertain {
        background: #f6f2ec;
        color: #756a5f;
        border-color: #ddd4c9;
    }

    .maturity-main {
        font-size: 1.08rem;
        font-weight: 700;
        color: #2b2723;
        margin-top: 0.15rem;
    }

    .maturity-confidence {
        margin-top: 0.35rem;
        color: #8a8179;
        font-size: 0.76rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>

    /* =======================================================
       BIGGER, MORE READABLE COCOATRACK TEXT
       ======================================================= */

    html, body, [class*="css"] {
        font-size: 17px !important;
    }

    /* Main headings */
    h1 {
        font-size: 2.25rem !important;
    }

    h2 {
        font-size: 1.85rem !important;
    }

    h3 {
        font-size: 1.45rem !important;
    }

    h4 {
        font-size: 1.20rem !important;
    }

    p {
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Metric labels */
    div[data-testid="stMetric"] label {
        font-size: 0.92rem !important;
        font-weight: 650 !important;
    }

    /* Metric values */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 750 !important;
    }

    /* Captions */
    [data-testid="stCaptionContainer"] {
        font-size: 0.90rem !important;
        line-height: 1.5 !important;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        font-size: 0.98rem !important;
        font-weight: 650 !important;
        min-height: 3rem !important;
    }

    /* Dataframe text */
    [data-testid="stDataFrame"] {
        font-size: 0.95rem !important;
    }

    /* Maturity cards */
    .maturity-card {
        padding: 1.15rem 1.2rem !important;
        min-height: 112px !important;
    }

    .maturity-pod-number {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #4e4842 !important;
    }

    .maturity-status {
        font-size: 1rem !important;
        font-weight: 750 !important;
        padding: 0.38rem 0.72rem !important;
    }

    .maturity-main {
        font-size: 1.22rem !important;
        font-weight: 750 !important;
    }

    .maturity-confidence {
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        color: #726a63 !important;
        margin-top: 0.5rem !important;
    }

    /* Stage description */
    .stage-copy h3 {
        font-size: 1.28rem !important;
    }

    .stage-copy p {
        font-size: 0.98rem !important;
    }

    /* Brand/header */
    .brand-title {
        font-size: 1.5rem !important;
    }

    .brand-subtitle {
        font-size: 0.95rem !important;
    }

    .status-pill {
        font-size: 0.92rem !important;
    }

    /* Form labels */
    label,
    .stRadio label,
    .stCheckbox label {
        font-size: 0.98rem !important;
    }

    /* Select boxes / inputs */
    input,
    textarea,
    [data-baseweb="select"] {
        font-size: 0.98rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def stage_heading(number: int, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="stage-box">
            <div class="stage-number">{number}</div>
            <div class="stage-copy">
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def to_png_bytes(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def ensure_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


# ============================================================
# STAGE 1 — QUALITY CHECK
# ============================================================

def check_input_quality(image_rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    height, width = image_rgb.shape[:2]
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    reasons = []

    if min(height, width) < 400:
        reasons.append("Low image resolution")

    if brightness < 35:
        reasons.append("Image may be too dark")

    if brightness > 235:
        reasons.append("Image may be overexposed")

    if blur_score < 45:
        reasons.append("Image may be blurry")

    status = "PASS" if not reasons else "CHECK"

    return {
        "status": status,
        "reasons": reasons,
        "metrics": {
            "Width": width,
            "Height": height,
            "Mean brightness": round(brightness, 2),
            "Contrast": round(contrast, 2),
            "Blur score": round(blur_score, 2),
        },
    }


# ============================================================
# STAGE 2 — WHITE CARDBOARD CROP
# ============================================================

def resize_long_side(image_rgb: np.ndarray, target_long_side: int = 1280):
    height, width = image_rgb.shape[:2]
    long_side = max(height, width)

    if long_side <= target_long_side:
        return image_rgb.copy(), 1.0

    scale = target_long_side / long_side
    new_size = (
        int(round(width * scale)),
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image_rgb,
        new_size,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def prepare_image(
    image_rgb: np.ndarray,
    target_long_side: int = 1280,
    paper_v_min: float = 0.82,
    paper_s_max: float = 0.24,
    minimum_paper_area: int = 5000,
    crop_margin: int = 15,
) -> dict:
    resized_rgb, scale = resize_long_side(
        image_rgb,
        target_long_side=target_long_side,
    )

    hsv = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    paper_pixel_mask = (
        (value >= paper_v_min) &
        (saturation <= paper_s_max)
    ).astype(np.uint8) * 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (25, 25),
    )

    clean_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    cleaned_paper_mask = cv2.morphologyEx(
        paper_pixel_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    cleaned_paper_mask = cv2.morphologyEx(
        cleaned_paper_mask,
        cv2.MORPH_OPEN,
        clean_kernel,
    )

    contours, _ = cv2.findContours(
        cleaned_paper_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError("Image area could not be detected.")

    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)

    if largest_area < minimum_paper_area:
        raise ValueError("Detected white region is too small to be the cardboard.")

    paper_hull = cv2.convexHull(largest_contour)

    full_roi_mask = np.zeros(
        resized_rgb.shape[:2],
        dtype=np.uint8,
    )

    cv2.drawContours(
        full_roi_mask,
        [paper_hull],
        contourIdx=-1,
        color=255,
        thickness=-1,
    )

    x, y, width, height = cv2.boundingRect(paper_hull)

    image_height, image_width = resized_rgb.shape[:2]

    x1 = max(x - crop_margin, 0)
    y1 = max(y - crop_margin, 0)
    x2 = min(x + width + crop_margin, image_width)
    y2 = min(y + height + crop_margin, image_height)

    cropped_rgb = resized_rgb[y1:y2, x1:x2].copy()
    cropped_roi_mask = full_roi_mask[y1:y2, x1:x2].copy()

    prepared_rgb = cv2.bitwise_and(
        cropped_rgb,
        cropped_rgb,
        mask=cropped_roi_mask,
    )

    overlay = resized_rgb.copy()
    overlay[full_roi_mask > 0] = (
        0.55 * overlay[full_roi_mask > 0] +
        0.45 * np.array([255, 80, 80])
    ).astype(np.uint8)

    return {
        "resized_rgb": resized_rgb,
        "paper_pixel_mask": paper_pixel_mask,
        "cleaned_paper_mask": cleaned_paper_mask,
        "paper_roi_mask": cropped_roi_mask,
        "prepared_rgb": prepared_rgb,
        "paper_overlay": overlay,
        "crop_box": (x1, y1, x2, y2),
        "scale": scale,
    }



# ============================================================
# RAW IMAGE — MANUAL ROI + GRABCUT
# ============================================================

def isolate_raw_pod_with_grabcut(
    image_rgb: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    iterations: int = 5,
) -> dict:
    """
    Isolate one pod from a raw image using a user-guided rectangle.

    The selected rectangle is expanded to include surrounding background.
    GrabCut then receives:
    - surrounding pixels as definite background,
    - the selected rectangle as probable foreground,
    - a smaller central ellipse as definite foreground.

    This is more reliable than running GrabCut directly on a tight box.
    """
    image_height, image_width = image_rgb.shape[:2]

    x = int(np.clip(x, 0, max(image_width - 2, 0)))
    y = int(np.clip(y, 0, max(image_height - 2, 0)))
    width = int(np.clip(width, 2, image_width - x))
    height = int(np.clip(height, 2, image_height - y))

    # Add real surrounding context so GrabCut has background examples.
    padding_x = max(12, int(width * 0.18))
    padding_y = max(12, int(height * 0.18))

    context_x1 = max(0, x - padding_x)
    context_y1 = max(0, y - padding_y)
    context_x2 = min(image_width, x + width + padding_x)
    context_y2 = min(image_height, y + height + padding_y)

    context_rgb = image_rgb[
        context_y1:context_y2,
        context_x1:context_x2,
    ].copy()

    context_height, context_width = context_rgb.shape[:2]

    inner_x = x - context_x1
    inner_y = y - context_y1

    preview = image_rgb.copy()

    cv2.rectangle(
        preview,
        (x, y),
        (x + width, y + height),
        (255, 220, 0),
        4,
    )

    cv2.putText(
        preview,
        "Selected pod region",
        (x, max(28, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 220, 0),
        2,
    )

    context_bgr = cv2.cvtColor(
        context_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # Start with definite background everywhere.
    grabcut_mask = np.full(
        (context_height, context_width),
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    # User rectangle becomes probable foreground.
    grabcut_mask[
        inner_y:inner_y + height,
        inner_x:inner_x + width,
    ] = cv2.GC_PR_FGD

    # A smaller central ellipse gives GrabCut a definite foreground seed.
    centre = (
        int(inner_x + width / 2),
        int(inner_y + height / 2),
    )

    axes = (
        max(3, int(width * 0.22)),
        max(3, int(height * 0.22)),
    )

    cv2.ellipse(
        grabcut_mask,
        centre,
        axes,
        0,
        0,
        360,
        cv2.GC_FGD,
        thickness=-1,
    )

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    used_fallback = False

    try:
        cv2.grabCut(
            context_bgr,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            int(iterations),
            cv2.GC_INIT_WITH_MASK,
        )

        context_foreground_mask = np.where(
            (grabcut_mask == cv2.GC_FGD)
            | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

    except cv2.error:
        context_foreground_mask = np.zeros(
            (context_height, context_width),
            dtype=np.uint8,
        )

        context_foreground_mask[
            inner_y:inner_y + height,
            inner_x:inner_x + width,
        ] = 255

        used_fallback = True

    # Return only the user's selected rectangle.
    crop_rgb = image_rgb[
        y:y + height,
        x:x + width,
    ].copy()

    crop_mask = context_foreground_mask[
        inner_y:inner_y + height,
        inner_x:inner_x + width,
    ].copy()

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    crop_mask = cv2.morphologyEx(
        crop_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    crop_mask = cv2.morphologyEx(
        crop_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    foreground_pixels = int(
        np.count_nonzero(crop_mask)
    )

    total_crop_pixels = int(
        crop_mask.size
    )

    foreground_percentage = (
        100.0 * foreground_pixels / total_crop_pixels
        if total_crop_pixels > 0
        else 0.0
    )

    # Prevent a useless 0% result. The user's rectangle is a valid manual
    # fallback when GrabCut cannot separate foreground from background.
    if foreground_percentage < 1.0:
        crop_mask = np.full(
            crop_rgb.shape[:2],
            255,
            dtype=np.uint8,
        )

        foreground_percentage = 100.0
        used_fallback = True

    isolated_rgb = cv2.bitwise_and(
        crop_rgb,
        crop_rgb,
        mask=crop_mask,
    )

    return {
        "preview": preview,
        "crop_rgb": crop_rgb,
        "crop_mask": crop_mask,
        "isolated_rgb": isolated_rgb,
        "crop_box": (x, y, x + width, y + height),
        "foreground_percentage": foreground_percentage,
        "used_fallback": used_fallback,
    }


# ============================================================
# STAGE 3 — GREEN-YELLOW + RED-BROWN MASKS
# ============================================================

def prepare_colour_masks(
    prepared_rgb: np.ndarray,
    paper_roi_mask: np.ndarray,
) -> dict:
    hsv = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2LAB)

    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)

    red = prepared_rgb[:, :, 0].astype(np.int16)
    green = prepared_rgb[:, :, 1].astype(np.int16)
    blue = prepared_rgb[:, :, 2].astype(np.int16)

    lab_a = lab[:, :, 1].astype(np.int16)

    inside_roi = paper_roi_mask > 0

    # Strong green-yellow support
    green_hue = (hue >= 16) & (hue <= 105)
    green_rgb = (
        (green >= red - 12) &
        (green >= blue - 10)
    )

    green_yellow_boolean = (
        green_hue &
        green_rgb &
        (saturation >= 24) &
        (value >= 15) &
        inside_roi
    )

    # Strong red-brown support
    red_low_hue = (hue >= 0) & (hue <= 48)
    red_high_hue = (hue >= 132) & (hue <= 179)

    red_rgb = (
        (red >= green - 10) &
        (red >= blue - 4) &
        (red >= 18)
    )

    red_lab = lab_a >= 128

    red_brown_boolean = (
        (
            (
                red_low_hue |
                red_high_hue
            ) &
            (saturation >= 10) &
            (value >= 8)
        ) |
        (
            red_rgb &
            red_lab &
            (value >= 8)
        )
    ) & inside_roi

    # Make masks mutually exclusive using stronger evidence
    strong_green = (
        (hue >= 25) &
        (hue <= 100) &
        (green >= red + 4) &
        (saturation >= 28)
    )

    strong_red = (
        red_high_hue |
        (lab_a >= 140) |
        (
            (red >= green + 12) &
            (red >= blue + 10)
        )
    )

    green_yellow_boolean &= ~strong_red
    red_brown_boolean &= ~strong_green

    green_yellow_mask = green_yellow_boolean.astype(np.uint8) * 255
    red_brown_mask = red_brown_boolean.astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    green_yellow_mask = cv2.morphologyEx(
        green_yellow_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    green_yellow_mask = cv2.morphologyEx(
        green_yellow_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    red_brown_mask = cv2.morphologyEx(
        red_brown_mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    red_brown_mask = cv2.morphologyEx(
        red_brown_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    green_yellow_mask = cv2.bitwise_and(
        green_yellow_mask,
        paper_roi_mask,
    )

    red_brown_mask = cv2.bitwise_and(
        red_brown_mask,
        paper_roi_mask,
    )

    combined_mask = cv2.bitwise_or(
        green_yellow_mask,
        red_brown_mask,
    )

    combined_rgb = cv2.bitwise_and(
        prepared_rgb,
        prepared_rgb,
        mask=combined_mask,
    )

    return {
        "hsv": hsv,
        "green_yellow_mask": green_yellow_mask,
        "red_brown_mask": red_brown_mask,
        "combined_mask": combined_mask,
        "combined_rgb": combined_rgb,
        "counts": {
            "Green-yellow pixels": int(np.count_nonzero(green_yellow_mask)),
            "Red-brown pixels": int(np.count_nonzero(red_brown_mask)),
            "Combined pixels": int(np.count_nonzero(combined_mask)),
        },
    }


# ============================================================
# STAGE 4 — ADJUSTABLE WATERSHED DETECTION
# ============================================================

def detect_pods_with_watershed(
    prepared_rgb: np.ndarray,
    combined_mask: np.ndarray,
    minimum_area: int,
    maximum_area: int,
    clean_opening_size: int,
    clean_closing_size: int,
    core_opening_size: int,
    core_distance: int,
    minimum_seed_area: int,
    minimum_solidity: float,
    minimum_aspect_ratio: float,
    maximum_aspect_ratio: float,
) -> dict:
    mask = combined_mask.astype(np.uint8)

    open_size = ensure_odd(clean_opening_size)
    close_size = ensure_odd(clean_closing_size)
    core_size = ensure_odd(core_opening_size)

    cleaned_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        ),
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        ),
    )

    thick_objects = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (core_size, core_size),
        ),
    )

    distance_map = cv2.distanceTransform(
        thick_objects,
        cv2.DIST_L2,
        5,
    )

    seed_mask = (
        distance_map >= float(core_distance)
    ).astype(np.uint8) * 255

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed_mask,
        connectivity=8,
    )

    filtered_seed_mask = np.zeros_like(seed_mask)

    for label_id in range(1, labels_count):
        seed_area = int(stats[label_id, cv2.CC_STAT_AREA])

        if seed_area >= int(minimum_seed_area):
            filtered_seed_mask[labels == label_id] = 255

    marker_count, markers = cv2.connectedComponents(
        filtered_seed_mask,
    )

    markers = markers.astype(np.int32) + 1

    background = cv2.bitwise_not(cleaned_mask)
    markers[background > 0] = 1

    unknown = cv2.subtract(
        cleaned_mask,
        filtered_seed_mask,
    )
    markers[unknown > 0] = 0

    watershed_input = cv2.cvtColor(
        prepared_rgb,
        cv2.COLOR_RGB2BGR,
    )

    watershed_markers = cv2.watershed(
        watershed_input,
        markers,
    )

    output_image = prepared_rgb.copy()

    accepted_mask = np.zeros_like(cleaned_mask)
    rejected_mask = np.zeros_like(cleaned_mask)

    accepted_regions = []
    rejected_regions = []

    for label_value in np.unique(watershed_markers):
        if label_value <= 1:
            continue

        region_mask = (
            watershed_markers == label_value
        ).astype(np.uint8) * 255

        region_mask = cv2.bitwise_and(
            region_mask,
            cleaned_mask,
        )

        contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        area = float(cv2.contourArea(contour))

        if area <= 0:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]

        long_side = max(rotated_width, rotated_height)
        short_side = min(rotated_width, rotated_height)

        aspect_ratio = (
            long_side / short_side
            if short_side > 0
            else 999.0
        )

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))

        solidity = (
            area / hull_area
            if hull_area > 0
            else 0.0
        )

        perimeter = float(cv2.arcLength(contour, True))

        circularity = (
            4.0 * np.pi * area / max(perimeter ** 2, 1.0)
        )

        accepted = (
            minimum_area <= area <= maximum_area
            and minimum_aspect_ratio <= aspect_ratio <= maximum_aspect_ratio
            and solidity >= minimum_solidity
        )

        region = {
            "contour": contour,
            "mask": region_mask,
            "bbox": (x, y, width, height),
            "area": area,
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "circularity": circularity,
            "accepted": accepted,
        }

        if accepted:
            accepted_regions.append(region)
            cv2.drawContours(
                accepted_mask,
                [contour],
                contourIdx=-1,
                color=255,
                thickness=-1,
            )
        else:
            rejected_regions.append(region)
            cv2.drawContours(
                rejected_mask,
                [contour],
                contourIdx=-1,
                color=255,
                thickness=-1,
            )

    for index, region in enumerate(accepted_regions, start=1):
        x, y, width, height = region["bbox"]

        cv2.rectangle(
            output_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3,
        )

        cv2.putText(
            output_image,
            f"Pod {index}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

    boundary_image = prepared_rgb.copy()
    boundary_image[watershed_markers == -1] = [255, 0, 255]

    return {
        "cleaned_mask": cleaned_mask,
        "thick_objects": thick_objects,
        "seed_mask": filtered_seed_mask,
        "boundary_image": boundary_image,
        "output_image": output_image,
        "accepted_mask": accepted_mask,
        "rejected_mask": rejected_mask,
        "accepted_regions": accepted_regions,
        "rejected_regions": rejected_regions,
        "seed_count": max(marker_count - 1, 0),
    }



def draw_manual_selection_overlay(
    prepared_rgb: np.ndarray,
    accepted_regions: list,
    selected_pod_ids: list,
) -> np.ndarray:
    """
    Draw selected pods in green and unselected pods in grey/red.
    Pod numbers match the Stage 4 detection order.
    """
    overlay = prepared_rgb.copy()
    selected_set = {int(pod_id) for pod_id in selected_pod_ids}

    for pod_id, region in enumerate(accepted_regions, start=1):
        contour = region["contour"]
        x, y, width, height = region["bbox"]

        is_selected = pod_id in selected_set

        if is_selected:
            box_colour = (0, 210, 0)
            text_colour = (0, 255, 0)
            status = "SELECTED"
            thickness = 4
        else:
            box_colour = (110, 110, 110)
            text_colour = (255, 80, 80)
            status = "OFF"
            thickness = 2

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            box_colour,
            thickness,
        )

        cv2.putText(
            overlay,
            f"Pod {pod_id} {status}",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            text_colour,
            2,
        )

    return overlay


# ============================================================
# STAGE 5 — LABEL PODS
# ============================================================

def label_pods(
    prepared_rgb: np.ndarray,
    accepted_regions: list,
) -> dict:
    """
    Label the exact locked pod regions selected by the user.

    The same bounding boxes from the Select page are carried into the
    Results page without rerunning detection or replacing them with
    newly generated regions.
    """
    label_image = np.zeros(
        prepared_rgb.shape[:2],
        dtype=np.int32,
    )

    overlay = prepared_rgb.copy()
    pods = []

    for pod_id, region in enumerate(
        accepted_regions,
        start=1,
    ):
        contour = np.asarray(
            region["contour"],
            dtype=np.int32,
        )

        x, y, width, height = [
            int(value)
            for value in region["bbox"]
        ]

        cv2.drawContours(
            label_image,
            [contour],
            contourIdx=-1,
            color=int(pod_id),
            thickness=-1,
        )

        # Reuse the exact selected bounding box.
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 200, 0),
            3,
        )

        label_text = f"Pod {pod_id}"

        cv2.putText(
            overlay,
            label_text,
            (
                x,
                max(24, y - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 200, 0),
            2,
        )

        pods.append(
            {
                "Pod_ID": pod_id,
                "bbox": (
                    x,
                    y,
                    width,
                    height,
                ),
                "contour": contour,
            }
        )

    return {
        "label_image": label_image,
        "overlay_ids": overlay,
        "pods": pods,
    }


# ============================================================
# STAGE 7 — MASK-OVERLAP CLASSIFICATION
# ============================================================

def extract_features(
    prepared_rgb: np.ndarray,
    label_image: np.ndarray,
    pods: list,
) -> pd.DataFrame:
    """
    Extract robust colour features from each detected cocoa pod.

    Improvements:
    - focuses on the inner portion of the pod mask
    - removes very dark, very bright and weak-colour pixels
    - keeps ordinary HSV means for compatibility
    - adds robust HSV medians and Lab-a median
    """

    hsv = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(prepared_rgb, cv2.COLOR_RGB2LAB)

    rows = []

    for pod in pods:
        pod_id = int(pod["Pod_ID"])

        full_mask = (
            label_image == pod_id
        ).astype(np.uint8)

        if np.count_nonzero(full_mask) == 0:
            continue

        contour = pod["contour"]

        area = float(
            cv2.contourArea(contour)
        )

        perimeter = float(
            cv2.arcLength(contour, True)
        )

        x, y, width, height = cv2.boundingRect(
            contour
        )

        rotated_width, rotated_height = (
            cv2.minAreaRect(contour)[1]
        )

        long_side = max(
            rotated_width,
            rotated_height,
        )

        short_side = min(
            rotated_width,
            rotated_height,
        )

        aspect_ratio = (
            long_side / short_side
            if short_side > 0
            else 0.0
        )

        hull = cv2.convexHull(contour)

        hull_area = float(
            cv2.contourArea(hull)
        )

        solidity = (
            area / hull_area
            if hull_area > 0
            else 0.0
        )

        circularity = (
            4.0
            * np.pi
            * area
            / max(perimeter ** 2, 1.0)
        )

        moments = cv2.moments(contour)

        centroid_x = (
            int(
                moments["m10"]
                / moments["m00"]
            )
            if moments["m00"] != 0
            else x + width // 2
        )

        centroid_y = (
            int(
                moments["m01"]
                / moments["m00"]
            )
            if moments["m00"] != 0
            else y + height // 2
        )

        # ----------------------------------------------------
        # INNER POD MASK
        #
        # Removes edge pixels, which are the pixels most
        # likely to contain leaves/background from imperfect
        # detector boundaries.
        # ----------------------------------------------------

        erosion_radius = max(
            1,
            min(
                5,
                int(
                    max(
                        1,
                        min(width, height)
                    )
                    * 0.06
                ),
            ),
        )

        kernel_size = (
            2 * erosion_radius + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        inner_mask = cv2.erode(
            full_mask,
            kernel,
            iterations=1,
        )

        # Fall back if erosion removed too much.
        if (
            np.count_nonzero(inner_mask)
            < 0.35
            * np.count_nonzero(full_mask)
        ):
            inner_mask = full_mask.copy()

        # ----------------------------------------------------
        # REMOVE LOW-QUALITY COLOUR PIXELS
        #
        # S < 25:
        #     weak / grey colour information
        #
        # V < 20:
        #     strong shadow
        #
        # V > 245:
        #     glare / clipped highlight
        # ----------------------------------------------------

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        colour_valid = (
            (inner_mask > 0)
            & (saturation >= 25)
            & (value >= 20)
            & (value <= 245)
        )

        valid_count = int(
            np.count_nonzero(colour_valid)
        )

        inner_count = int(
            np.count_nonzero(inner_mask)
        )

        # If filtering became too aggressive, use the inner
        # mask rather than throwing the pod away.
        if valid_count < max(
            30,
            int(inner_count * 0.20),
        ):
            colour_valid = (
                inner_mask > 0
            )

            valid_count = int(
                np.count_nonzero(
                    colour_valid
                )
            )

        hsv_pixels = hsv[
            colour_valid
        ]

        lab_pixels = lab[
            colour_valid
        ]

        # Original whole-pod means retained for compatibility.
        whole_hsv_pixels = hsv[
            full_mask > 0
        ]

        h_mean = float(
            np.mean(
                whole_hsv_pixels[:, 0]
            )
        )

        s_mean = float(
            np.mean(
                whole_hsv_pixels[:, 1]
            )
        )

        v_mean = float(
            np.mean(
                whole_hsv_pixels[:, 2]
            )
        )

        # Robust colour representation.
        h_robust = float(
            np.median(
                hsv_pixels[:, 0]
            )
        )

        s_robust = float(
            np.median(
                hsv_pixels[:, 1]
            )
        )

        v_robust = float(
            np.median(
                hsv_pixels[:, 2]
            )
        )

        lab_a_robust = float(
            np.median(
                lab_pixels[:, 1]
            )
        )

        valid_fraction = (
            valid_count
            / max(inner_count, 1)
        )

        rows.append(
            {
                "Pod_ID": pod_id,
                "Area": round(
                    area,
                    2,
                ),
                "Aspect_Ratio": round(
                    aspect_ratio,
                    4,
                ),
                "Solidity": round(
                    solidity,
                    4,
                ),
                "Circularity": round(
                    circularity,
                    4,
                ),

                # Original means
                "H_Mean": round(
                    h_mean,
                    4,
                ),
                "S_Mean": round(
                    s_mean,
                    4,
                ),
                "V_Mean": round(
                    v_mean,
                    4,
                ),

                # New robust maturity features
                "H_Robust": round(
                    h_robust,
                    4,
                ),
                "S_Robust": round(
                    s_robust,
                    4,
                ),
                "V_Robust": round(
                    v_robust,
                    4,
                ),
                "LabA_Robust": round(
                    lab_a_robust,
                    4,
                ),
                "Valid_Colour_Fraction": round(
                    valid_fraction,
                    4,
                ),

                "CentroidX": centroid_x,
                "CentroidY": centroid_y,
                "X": x,
                "Y": y,
                "Width": width,
                "Height": height,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STAGE 7 — MASK-OVERLAP CLASSIFICATION
# ============================================================


def classify_by_mask_overlap(
    features_df: pd.DataFrame,
    label_image: np.ndarray,
    green_yellow_mask: np.ndarray,
    red_brown_mask: np.ndarray,
    minimum_margin_percentage: float = 5.0,
    minimum_evidence_coverage: float = 5.0,
) -> pd.DataFrame:
    output = features_df.copy()

    if output.empty:
        output.attrs["image_majority"] = "UNCERTAIN"
        output.attrs["green_candidate_count"] = 0
        output.attrs["red_candidate_count"] = 0
        return output

    rows = []

    for _, feature_row in output.iterrows():
        pod_id = int(feature_row["Pod_ID"])

        full_pod_mask = (
            label_image == pod_id
        )

        x = int(feature_row["X"])
        y = int(feature_row["Y"])
        width = int(feature_row["Width"])
        height = int(feature_row["Height"])

        inset_x = max(
            1,
            int(width * 0.15),
        )

        inset_y = max(
            1,
            int(height * 0.15),
        )

        inner_x1 = min(
            label_image.shape[1] - 1,
            x + inset_x,
        )

        inner_y1 = min(
            label_image.shape[0] - 1,
            y + inset_y,
        )

        inner_x2 = max(
            inner_x1 + 1,
            min(
                label_image.shape[1],
                x + width - inset_x,
            ),
        )

        inner_y2 = max(
            inner_y1 + 1,
            min(
                label_image.shape[0],
                y + height - inset_y,
            ),
        )

        inner_box_mask = np.zeros(
            full_pod_mask.shape,
            dtype=bool,
        )

        inner_box_mask[
            inner_y1:inner_y2,
            inner_x1:inner_x2
        ] = True

        pod_mask = (
            full_pod_mask
            & inner_box_mask
        )

        # Fallback for tiny detections.
        if np.count_nonzero(pod_mask) < 10:
            pod_mask = full_pod_mask

        pod_pixels = int(
            np.count_nonzero(
                pod_mask
            )
        )

        green_overlap = int(
            np.count_nonzero(
                (green_yellow_mask > 0) &
                pod_mask
            )
        )

        red_overlap = int(
            np.count_nonzero(
                (red_brown_mask > 0) &
                pod_mask
            )
        )

        green_percentage = (
            100.0 * green_overlap / pod_pixels
            if pod_pixels > 0
            else 0.0
        )

        red_percentage = (
            100.0 * red_overlap / pod_pixels
            if pod_pixels > 0
            else 0.0
        )

        evidence_coverage = green_percentage + red_percentage
        margin = abs(green_percentage - red_percentage)

        if evidence_coverage < minimum_evidence_coverage:
            initial_colour = "UNCERTAIN"

        elif margin < minimum_margin_percentage:
            initial_colour = "UNCERTAIN"

        elif green_percentage > red_percentage:
            initial_colour = "GREEN-YELLOW"

        else:
            initial_colour = "RED-BROWN"

        row = feature_row.to_dict()
        row.update(
            {
                "Green_Pixels": green_overlap,
                "Red_Brown_Pixels": red_overlap,
                "Green_Percentage": round(green_percentage, 2),
                "Red_Brown_Percentage": round(red_percentage, 2),
                "Colour_Evidence_Coverage": round(evidence_coverage, 2),
                "Score_Margin": round(margin, 2),
                "Initial_Colour": initial_colour,
            }
        )

        rows.append(row)

    classified = pd.DataFrame(rows)

    green_count = int(
        (classified["Initial_Colour"] == "GREEN-YELLOW").sum()
    )

    red_count = int(
        (classified["Initial_Colour"] == "RED-BROWN").sum()
    )

    if green_count == 0 and red_count == 0:
        image_majority = "UNCERTAIN"
    elif green_count >= red_count:
        image_majority = "GREEN-YELLOW"
    else:
        image_majority = "RED-BROWN"

    classified["Image_Majority"] = image_majority

    # Final class follows accepted-candidate majority.
    classified["Pod_Color"] = np.where(
        classified["Initial_Colour"] == "UNCERTAIN",
        "UNCERTAIN",
        image_majority,
    )

    classified.attrs["image_majority"] = image_majority
    classified.attrs["green_candidate_count"] = green_count
    classified.attrs["red_candidate_count"] = red_count

    return classified



# ============================================================
# MATURITY CLASSIFICATION — HSV REFERENCE PROFILES
# ============================================================

MATURITY_REFERENCE = {
    "UNRIPE": {
        "H_mean": 47.115346,
        "H_std": 4.132668,
        "S_mean": 110.453087,
        "S_std": 14.635553,
        "V_mean": 99.705481,
        "V_std": 17.088032,
    },
    "HALF-RIPE": {
        "H_mean": 39.099117,
        "H_std": 4.309698,
        "S_mean": 112.585117,
        "S_std": 35.986978,
        "V_mean": 114.969550,
        "V_std": 22.157898,
    },
    "RIPE": {
        "H_mean": 33.637330,
        "H_std": 6.173111,
        "S_mean": 184.843300,
        "S_std": 19.296800,
        "V_mean": 99.146830,
        "V_std": 20.522517,
    },
}


def classify_maturity_from_hsv(
    classified_df: pd.DataFrame,
    uncertainty_threshold: float = 0.46,
) -> pd.DataFrame:
    """
    Robust experimental maturity estimation.

    Uses:
    - robust HSV medians from inner pod pixels
    - Lab-a red/green evidence
    - green/red-brown pixel composition
    - maturity-specific feature weighting

    Reference profiles remain derived from the available
    labelled cocoa-pod measurements.
    """

    output = classified_df.copy()

    if output.empty:
        output.attrs[
            "maturity_majority"
        ] = "UNCERTAIN"

        return output

    result_rows = []

    for _, row in output.iterrows():

        # Prefer robust measurements.
        h_value = float(
            row.get(
                "H_Robust",
                row["H_Mean"],
            )
        )

        s_value = float(
            row.get(
                "S_Robust",
                row["S_Mean"],
            )
        )

        v_value = float(
            row.get(
                "V_Robust",
                row["V_Mean"],
            )
        )

        lab_a = float(
            row.get(
                "LabA_Robust",
                128.0,
            )
        )

        valid_fraction = float(
            row.get(
                "Valid_Colour_Fraction",
                1.0,
            )
        )

        green_percentage = float(
            row.get(
                "Green_Percentage",
                0.0,
            )
        )

        red_percentage = float(
            row.get(
                "Red_Brown_Percentage",
                0.0,
            )
        )

        # ----------------------------------------------------
        # REFERENCE MIDPOINTS DERIVED DIRECTLY FROM THE
        # EXISTING DATASET PROFILES
        # ----------------------------------------------------

        unripe = MATURITY_REFERENCE[
            "UNRIPE"
        ]

        half = MATURITY_REFERENCE[
            "HALF-RIPE"
        ]

        ripe = MATURITY_REFERENCE[
            "RIPE"
        ]

        h_unripe_half = (
            unripe["H_mean"]
            + half["H_mean"]
        ) / 2.0

        h_half_ripe = (
            half["H_mean"]
            + ripe["H_mean"]
        ) / 2.0

        s_half_ripe = (
            half["S_mean"]
            + ripe["S_mean"]
        ) / 2.0

        # ----------------------------------------------------
        # CLASS-SPECIFIC DISTANCES
        #
        # UNRIPE/HALF-RIPE:
        # Hue is much more informative than saturation.
        #
        # RIPE:
        # Saturation becomes highly informative in the
        # available reference data.
        # ----------------------------------------------------

        distances = {}

        for maturity_class, profile in (
            MATURITY_REFERENCE.items()
        ):

            h_z = (
                h_value
                - profile["H_mean"]
            ) / max(
                profile["H_std"],
                1e-6,
            )

            s_z = (
                s_value
                - profile["S_mean"]
            ) / max(
                profile["S_std"],
                1e-6,
            )

            v_z = (
                v_value
                - profile["V_mean"]
            ) / max(
                profile["V_std"],
                1e-6,
            )

            if maturity_class in (
                "UNRIPE",
                "HALF-RIPE",
            ):
                # S is weak for separating these two
                # classes in the available reference data.
                distance = np.sqrt(
                    0.70 * h_z ** 2
                    + 0.10 * s_z ** 2
                    + 0.20 * v_z ** 2
                )

            else:
                # Ripe pods show substantially stronger
                # saturation separation.
                distance = np.sqrt(
                    0.40 * h_z ** 2
                    + 0.50 * s_z ** 2
                    + 0.10 * v_z ** 2
                )

            distances[
                maturity_class
            ] = float(distance)

        class_names = list(
            distances.keys()
        )

        distance_values = np.array(
            [
                distances[name]
                for name in class_names
            ],
            dtype=np.float64,
        )

        similarity = np.exp(
            -(
                distance_values
                - distance_values.min()
            )
        )

        # ----------------------------------------------------
        # COLOUR-COMPOSITION EVIDENCE
        # ----------------------------------------------------

        unripe_index = (
            class_names.index("UNRIPE")
        )

        half_index = (
            class_names.index(
                "HALF-RIPE"
            )
        )

        ripe_index = (
            class_names.index("RIPE")
        )

        # Strong green evidence supports unripe.
        if (
            green_percentage
            >= red_percentage + 15.0
        ):
            similarity[
                unripe_index
            ] *= 1.18

        # Mixed colour evidence supports transition stage.
        if (
            green_percentage >= 10.0
            and red_percentage >= 10.0
            and abs(
                green_percentage
                - red_percentage
            ) <= 30.0
        ):
            similarity[
                half_index
            ] *= 1.20

        # Strong red/brown evidence supports ripe.
        if (
            red_percentage
            >= green_percentage + 15.0
        ):
            similarity[
                ripe_index
            ] *= 1.22

        # Lab-a: larger values indicate stronger red
        # tendency. Threshold 140 is already used by
        # CocoaTrack's strong-red mask.
        if lab_a >= 140.0:
            similarity[
                ripe_index
            ] *= 1.18

        # ----------------------------------------------------
        # STRONG RIPE EVIDENCE
        #
        # Dataset:
        # half-ripe S ≈ 112.6
        # ripe S ≈ 184.8
        #
        # Therefore saturation above their midpoint,
        # combined with low hue/red evidence, is strong
        # evidence for ripe colour.
        # ----------------------------------------------------

        strong_ripe = (
            s_value >= s_half_ripe
            and (
                h_value <= h_half_ripe
                or lab_a >= 140.0
                or (
                    red_percentage
                    >= green_percentage
                    + 10.0
                )
            )
        )

        # Hue above the unripe/half midpoint is
        # supportive of unripe colour.
        strong_unripe = (
            h_value >= h_unripe_half
            and green_percentage
            >= red_percentage
        )

        if strong_ripe:
            similarity[
                ripe_index
            ] *= 1.35

        elif strong_unripe:
            similarity[
                unripe_index
            ] *= 1.15

        probabilities = (
            similarity
            / max(
                similarity.sum(),
                1e-12,
            )
        )

        best_index = int(
            np.argmax(
                probabilities
            )
        )

        predicted_class = (
            class_names[
                best_index
            ]
        )

        confidence = float(
            probabilities[
                best_index
            ]
        )

        # Poor usable colour coverage makes the
        # maturity estimate less trustworthy.
        if valid_fraction < 0.12:
            final_class = "UNCERTAIN"

        elif (
            confidence
            < uncertainty_threshold
        ):
            final_class = "UNCERTAIN"

        else:
            final_class = (
                predicted_class
            )

        updated_row = (
            row.to_dict()
        )

        updated_row.update(
            {
                "Distance_Unripe":
                    round(
                        distances[
                            "UNRIPE"
                        ],
                        4,
                    ),

                "Distance_Half_Ripe":
                    round(
                        distances[
                            "HALF-RIPE"
                        ],
                        4,
                    ),

                "Distance_Ripe":
                    round(
                        distances[
                            "RIPE"
                        ],
                        4,
                    ),

                "Unripe_Score":
                    round(
                        float(
                            probabilities[
                                unripe_index
                            ]
                        ),
                        4,
                    ),

                "Half_Ripe_Score":
                    round(
                        float(
                            probabilities[
                                half_index
                            ]
                        ),
                        4,
                    ),

                "Ripe_Score":
                    round(
                        float(
                            probabilities[
                                ripe_index
                            ]
                        ),
                        4,
                    ),

                "Maturity_Confidence":
                    round(
                        confidence,
                        4,
                    ),

                "Maturity_Class":
                    final_class,

                "Strong_Ripe_Evidence":
                    bool(
                        strong_ripe
                    ),
            }
        )

        result_rows.append(
            updated_row
        )

    result = pd.DataFrame(
        result_rows
    )

    certain_predictions = result.loc[
        result[
            "Maturity_Class"
        ] != "UNCERTAIN",
        "Maturity_Class",
    ]

    if certain_predictions.empty:
        maturity_majority = (
            "UNCERTAIN"
        )

    else:
        counts = (
            certain_predictions
            .value_counts()
        )

        if (
            len(counts) > 1
            and counts.iloc[0]
            == counts.iloc[1]
        ):
            maturity_majority = (
                "MIXED"
            )

        else:
            maturity_majority = str(
                counts.index[0]
            )

    result[
        "Image_Maturity_Majority"
    ] = maturity_majority

    result.attrs[
        "maturity_majority"
    ] = maturity_majority

    return result


# ============================================================
# SESSION STATE DEFAULTS
# ============================================================


if "continue_to_classification" not in st.session_state:
    st.session_state.continue_to_classification = False


def reset_classification() -> None:
    """Clear any previous pod selection when the analysis changes."""
    st.session_state.continue_to_classification = False
    st.session_state.pop("selected_pod_ids", None)
    for key in [
        "locked_selected_regions",
        "locked_selected_pod_ids",
        "locked_prepared_rgb",
        "locked_green_yellow_mask",
        "locked_red_brown_mask",
        "locked_selection_overlay",
    ]:
        st.session_state.pop(key, None)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="brand-group">
            <div class="brand-mark">🍫</div>
            <div>
                <div class="brand-title">CocoaTrack</div>
                <div class="brand-subtitle">
                    Cocoa pod detection, colour analysis, and maturity estimation
                </div>
            </div>
        </div>
        <div class="header-badge">Analysis workspace</div>
    </div>
    """,
    unsafe_allow_html=True,
)





st.markdown(
    """
    <style>

    /* =======================================================
       COCOATRACK NAVIGATION
       ======================================================= */

    div[data-testid="stRadio"] > div {
        gap: 1rem !important;
        background: transparent !important;
    }

    div[data-testid="stRadio"] label {
        position: relative !important;
        min-width: 185px !important;
        min-height: 72px !important;
        padding: 0 !important;
        border: 1px solid #ded6cc !important;
        border-radius: 18px !important;
        background: #ffffff !important;
        box-shadow: 0 8px 20px rgba(70, 48, 33, 0.06) !important;
        transition:
            transform 0.18s ease,
            border-color 0.18s ease,
            box-shadow 0.18s ease,
            background 0.18s ease !important;
        cursor: pointer !important;
    }

    div[data-testid="stRadio"] label:hover {
        transform: translateY(-3px) !important;
        border-color: #9a725a !important;
        box-shadow: 0 12px 26px rgba(70, 48, 33, 0.12) !important;
        background: #fffdfa !important;
    }

    div[data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] label > div:last-child {
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 1rem 1.35rem !important;
        font-size: 1.12rem !important;
        font-weight: 750 !important;
        color: #3b332d !important;
        letter-spacing: -0.01em !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: #5b3a29 !important;
        border-color: #5b3a29 !important;
        box-shadow: 0 10px 24px rgba(70, 48, 33, 0.18) !important;
        transform: translateY(-2px) !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) > div:last-child {
        color: #ffffff !important;
    }

    /* Add simple visual symbols without emoji styling */
    div[data-testid="stRadio"] label:nth-of-type(1) > div:last-child::before {
        content: "⌂";
        font-size: 1.35rem;
        margin-right: 0.55rem;
        font-weight: 700;
    }

    div[data-testid="stRadio"] label:nth-of-type(2) > div:last-child::before {
        content: "◎";
        font-size: 1.28rem;
        margin-right: 0.55rem;
        font-weight: 700;
    }

    @media (max-width: 700px) {
        div[data-testid="stRadio"] > div {
            flex-direction: column !important;
        }

        div[data-testid="stRadio"] label {
            width: 100% !important;
            min-width: 0 !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>

    /* =======================================================
       FINAL UI CLEANUP
       ======================================================= */

    /* Give the whole app more breathing room */
    .block-container {
        max-width: 1180px !important;
        padding-top: 1.6rem !important;
        padding-bottom: 3rem !important;
    }


    /* -------------------------------------------------------
       TOP NAVIGATION — actual Streamlit buttons
       ------------------------------------------------------- */

    button[kind="primary"],
    button[kind="secondary"] {
        border-radius: 12px !important;
        min-height: 48px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        box-shadow: none !important;
        transition:
            transform 0.15s ease,
            border-color 0.15s ease !important;
    }

    button[kind="secondary"] {
        background: #ffffff !important;
        color: #4a4039 !important;
        border: 1px solid #ddd5cd !important;
    }

    button[kind="secondary"]:hover {
        border-color: #80604d !important;
        transform: translateY(-1px) !important;
    }

    button[kind="primary"] {
        background: #62402e !important;
        color: #ffffff !important;
        border: 1px solid #62402e !important;
    }

    button[kind="primary"]:hover {
        background: #513425 !important;
        border-color: #513425 !important;
        transform: translateY(-1px) !important;
    }


    /* -------------------------------------------------------
       RESET RADIO BUTTONS
       Previous CSS accidentally turned every radio into
       enormous navigation cards.
       ------------------------------------------------------- */

    div[data-testid="stRadio"] {
        background: transparent !important;
    }

    div[data-testid="stRadio"] > div {
        gap: 0.55rem !important;
        background: transparent !important;
    }

    div[data-testid="stRadio"] label {
        min-width: 0 !important;
        min-height: 0 !important;
        width: auto !important;
        height: auto !important;

        padding: 0.7rem 1rem !important;

        border: 1px solid #ddd6cf !important;
        border-radius: 10px !important;

        background: #ffffff !important;
        box-shadow: none !important;

        transform: none !important;
    }

    div[data-testid="stRadio"] label:hover {
        transform: none !important;
        background: #faf8f5 !important;
        border-color: #a68a78 !important;
        box-shadow: none !important;
    }

    /* Restore normal radio circle */
    div[data-testid="stRadio"] label > div:first-child {
        display: flex !important;
    }

    div[data-testid="stRadio"] label > div:last-child {
        width: auto !important;
        height: auto !important;

        display: block !important;

        padding: 0 !important;

        font-size: 0.98rem !important;
        font-weight: 600 !important;

        color: #443c36 !important;
    }

    /* Stop pseudo-icons injected by previous nav styling */
    div[data-testid="stRadio"] label > div:last-child::before {
        content: none !important;
        display: none !important;
    }

    div[data-testid="stRadio"] label:has(input:checked) {
        background: #f3ece6 !important;
        border-color: #8a6853 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    div[data-testid="stRadio"] label:has(input:checked)
    > div:last-child {
        color: #513728 !important;
    }


    /* -------------------------------------------------------
       SECTION / STEP CARDS
       ------------------------------------------------------- */

    .stage-card,
    .journey-card,
    .step-card {
        box-shadow: none !important;
    }


    /* Less giant vertical whitespace */
    div[data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }


    /* -------------------------------------------------------
       ANALYSIS PATH
       ------------------------------------------------------- */

    .analysis-path {
        background: #f4f0eb !important;
        border: 1px solid #e7dfd7 !important;
        border-radius: 10px !important;

        padding: 0.75rem 1rem !important;
        margin: 1rem 0 1.2rem 0 !important;

        color: #71665e !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;

        text-align: left !important;
    }


    /* -------------------------------------------------------
       HEADINGS
       ------------------------------------------------------- */

    h1, h2, h3, h4 {
        letter-spacing: -0.025em !important;
    }

    h2 {
        margin-top: 1.1rem !important;
    }


    /* -------------------------------------------------------
       GENERIC CARDS
       ------------------------------------------------------- */

    .feature-card,
    .ct-info-card,
    .ct-method-card,
    .maturity-card {
        box-shadow: none !important;
        border-color: #e5ded7 !important;
    }


    /* -------------------------------------------------------
       REMOVE EXCESSIVE ICON-LIKE VISUAL WEIGHT
       ------------------------------------------------------- */

    .feature-icon {
        font-size: 1.15rem !important;
        margin-bottom: 0.45rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>

    /* =======================================================
       GLOBAL TYPOGRAPHY — ROBOTO MEDIUM / BOLD
       ======================================================= */

    html,
    body,
    .stApp,
    .stApp * {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 600 !important;
    }

    /* Main headings */
    h1,
    h2,
    h3,
    h4,
    h5,
    h6 {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 700 !important;
    }

    /* Buttons */
    button,
    .stButton button,
    .stDownloadButton button {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 700 !important;
    }

    /* Inputs / selectors */
    input,
    textarea,
    select,
    label,
    [data-baseweb="select"],
    [data-baseweb="input"] {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 600 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 700 !important;
    }

    /* Metrics */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
    }

    /* Captions and helper text */
    [data-testid="stCaptionContainer"],
    .stCaption {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 600 !important;
    }

    /* Markdown */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;
    }

    /* Radio and checkbox text */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 600 !important;
    }

    /* Dataframes / tables */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] * {
        font-family:
            "Roboto",
            "Arial",
            "Helvetica",
            sans-serif !important;

        font-weight: 600 !important;
    }

    /* Custom CocoaTrack classes */
    .ct-home-title,
    .ct-card-title,
    .ct-method-value,
    .ct-flow-label,
    .maturity-status,
    .maturity-main,
    .feature-title,
    .stage-title {
        font-weight: 700 !important;
    }

    .ct-home-description,
    .ct-card-text,
    .ct-method-label,
    .ct-flow-num,
    .maturity-confidence,
    .maturity-pod-number,
    .analysis-path,
    .stage-copy,
    .feature-text {
        font-weight: 600 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# TOP NAVIGATION
# ============================================================

if "main_nav" not in st.session_state:
    st.session_state.main_nav = "Home"

# If an old session was left on the removed Raw Image page,
# return it safely to Analyze.
if st.session_state.get("main_nav") == "Raw Image":
    st.session_state.main_nav = "Analyze"

# Recover old sessions that were left on removed pages.
if st.session_state.get("main_nav") in {
    "Raw Image",
    "Learn",
}:
    st.session_state.main_nav = "Home"

# Clean page navigation.
nav_home, nav_analyze, nav_space = st.columns(
    [1, 1, 6],
    gap="small",
)

with nav_home:
    if st.button(
        "Home",
        key="nav_home_button",
        type=(
            "primary"
            if st.session_state.main_nav == "Home"
            else "secondary"
        ),
        width="stretch",
    ):
        st.session_state.main_nav = "Home"
        st.rerun()

with nav_analyze:
    if st.button(
        "Analyze",
        key="nav_analyze_button",
        type=(
            "primary"
            if st.session_state.main_nav == "Analyze"
            else "secondary"
        ),
        width="stretch",
    ):
        st.session_state.main_nav = "Analyze"
        st.rerun()

navigation_page = st.session_state.main_nav


if navigation_page == "Home":

    st.markdown(
        """
        <style>
        .home-main-card {
            background: #ffffff;
            border: 1px solid #e5ddd6;
            border-radius: 20px;
            padding: 2.8rem 2.5rem;
            margin-top: 2rem;
            margin-bottom: 1.4rem;
            text-align: center;
        }

        .home-kicker {
            color: #8a624c;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .home-main-title {
            color: #302722;
            font-size: 3.25rem;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 0.8rem;
        }

        .home-main-subtitle {
            color: #756b64;
            font-size: 1.08rem;
            line-height: 1.55;
            font-weight: 500;
            max-width: 720px;
            margin: 0 auto;
        }

        .home-feature-card {
            background: #ffffff;
            border: 1px solid #e5ddd6;
            border-radius: 15px;
            padding: 1.25rem 1rem;
            min-height: 112px;
            text-align: center;
        }

        .home-feature-number {
            color: #9b7560;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .home-feature-title {
            color: #372c26;
            font-size: 1.05rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    hero_html = (
        '<div class="home-main-card">'
        '<div class="home-kicker">Computer Vision Prototype</div>'
        '<div class="home-main-title">CocoaTrack</div>'
        '<div class="home-main-subtitle">'
        'Detect cocoa pods, estimate maturity, and predict harvestable dry bean yield from field images.'
        '</div>'
        '</div>'
    )

    st.markdown(
        hero_html,
        unsafe_allow_html=True,
    )

    feature_1, feature_2, feature_3 = st.columns(
        3,
        gap="medium",
    )

    with feature_1:
        st.markdown(
            '<div class="home-feature-card">'
            '<div class="home-feature-number">01</div>'
            '<div class="home-feature-title">Detect Cocoa Pods</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with feature_2:
        st.markdown(
            '<div class="home-feature-card">'
            '<div class="home-feature-number">02</div>'
            '<div class="home-feature-title">Estimate Maturity</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with feature_3:
        st.markdown(
            '<div class="home-feature-card">'
            '<div class="home-feature-number">03</div>'
            '<div class="home-feature-title">Estimate Yield</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    button_left, button_center, button_right = st.columns(
        [3, 2.2, 3]
    )

    with button_center:
        if st.button(
            "Start Analysis →",
            key="balanced_home_start",
            type="primary",
            width="stretch",
        ):
            st.session_state.main_nav = "Analyze"
            st.rerun()

    st.stop()

    # ========================================================
    # COCOATRACK HOME
    # ========================================================

    st.markdown(
        """
        <style>

        .ct-home-hero {
            background: #ffffff;
            border: 1px solid #e3ddd5;
            border-radius: 16px;
            padding: 2.4rem 2.5rem;
            margin-top: 0.8rem;
            margin-bottom: 1.8rem;
        }

        .ct-home-kicker {
            color: #795845;
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .ct-home-title {
            color: #2b2723;
            font-size: 2.55rem;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -0.035em;
            margin-bottom: 0.85rem;
        }

        .ct-home-description {
        width: 100%;
        max-width: none;
        display: block;
        margin-top: 22px;
        color: #6f6762;
        font-size: 1.06rem;
        line-height: 1.75;
        font-weight: 500;
    }

        .ct-section-title {
            color: #2b2723;
            font-size: 1.4rem;
            font-weight: 750;
            margin-top: 1.5rem;
            margin-bottom: 0.9rem;
        }

        .ct-info-card {
            background: #ffffff;
            border: 1px solid #e3ddd5;
            border-radius: 12px;
            padding: 1.25rem 1.3rem;
            min-height: 170px;
        }

        .ct-card-number {
            color: #8a6a56;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .ct-card-title {
            color: #2b2723;
            font-size: 1.12rem;
            font-weight: 750;
            margin-bottom: 0.5rem;
        }

        .ct-card-text {
            color: #756e67;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        .ct-flow {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.55rem;
            margin-top: 0.7rem;
        }

        .ct-flow-step {
            background: #ffffff;
            border: 1px solid #e3ddd5;
            border-radius: 10px;
            padding: 1rem 0.75rem;
            text-align: center;
        }

        .ct-flow-num {
            color: #5b3a29;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .ct-flow-label {
            color: #39332e;
            font-size: 0.9rem;
            font-weight: 650;
            line-height: 1.35;
        }

        .ct-method-card {
            background: #faf8f5;
            border: 1px solid #e5dfd7;
            border-radius: 12px;
            padding: 1.15rem 1.2rem;
            min-height: 130px;
        }

        .ct-method-label {
            color: #8a8179;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.45rem;
        }

        .ct-method-value {
            color: #302a26;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.45;
        }

        .ct-scope {
            background: #ffffff;
            border: 1px solid #e3ddd5;
            border-left: 4px solid #8c6b55;
            border-radius: 10px;
            padding: 1rem 1.15rem;
            color: #6d655f;
            font-size: 0.93rem;
            line-height: 1.55;
            margin-top: 1.2rem;
        }

        @media (max-width: 900px) {
            .ct-home-title {
                font-size: 2rem;
            }

            .ct-home-hero {
                padding: 1.6rem;
            }

            .ct-flow {
                grid-template-columns: 1fr;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    home_hero_html = (
        '<div class="ct-home-hero">'
        '<div class="ct-home-kicker">Computer Vision Prototype</div>'
        '<div class="ct-home-title">'
        'Cocoa Pod Detection, Maturity Estimation &amp; Yield Prediction'
        '</div>'
        '<div class="ct-home-description">'
        'CocoaTrack analyses cocoa field images by detecting cocoa pods, '
        'refining each detected pod region, estimating maturity from colour '
        'information, and converting detected ripe pods into an estimated '
        'dry cocoa bean yield.'
        '</div>'
        '</div>'
    )

    st.markdown(
        home_hero_html,
        unsafe_allow_html=True,
    )

    if st.button(
        "Analyze Cocoa Image",
        type="primary",
        width="stretch",
    ):
        st.session_state.main_nav = "Analyze"
        st.rerun()

    # --------------------------------------------------------
    # WHAT IT DOES
    # --------------------------------------------------------

    st.markdown(
        '<div class="ct-section-title">What CocoaTrack Does</div>',
        unsafe_allow_html=True,
    )

    capability_1, capability_2, capability_3 = st.columns(
        3,
        gap="medium",
    )

    with capability_1:
        st.markdown(
            """
            <div class="ct-info-card">
                <div class="ct-card-number">01</div>
                <div class="ct-card-title">Detect Cocoa Pods</div>
                <div class="ct-card-text">
                    A trained Edge Impulse YOLO-Pro model identifies
                    cocoa pods within the selected image.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with capability_2:
        st.markdown(
            """
            <div class="ct-info-card">
                <div class="ct-card-number">02</div>
                <div class="ct-card-title">Estimate Pod Maturity</div>
                <div class="ct-card-text">
                    Detected pod regions are refined before HSV,
                    Lab and colour-composition features are used
                    to estimate unripe, half-ripe, ripe or
                    uncertain maturity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with capability_3:
        st.markdown(
            """
            <div class="ct-info-card">
                <div class="ct-card-number">03</div>
                <div class="ct-card-title">Estimate Yield</div>
                <div class="ct-card-text">
                    Pods estimated as ripe are converted to an
                    estimated dry cocoa bean yield using the
                    prototype pod-index approach.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown(
        '<div class="ct-section-title">Analysis Workflow</div>',
        unsafe_allow_html=True,
    )

    workflow_html = (
        '<div class="ct-flow">'
        '<div class="ct-flow-step">'
        '<div class="ct-flow-num">01</div>'
        '<div class="ct-flow-label">Choose or upload image</div>'
        '</div>'
        '<div class="ct-flow-step">'
        '<div class="ct-flow-num">02</div>'
        '<div class="ct-flow-label">Detect cocoa pods</div>'
        '</div>'
        '<div class="ct-flow-step">'
        '<div class="ct-flow-num">03</div>'
        '<div class="ct-flow-label">Refine pod regions</div>'
        '</div>'
        '<div class="ct-flow-step">'
        '<div class="ct-flow-num">04</div>'
        '<div class="ct-flow-label">Estimate maturity</div>'
        '</div>'
        '<div class="ct-flow-step">'
        '<div class="ct-flow-num">05</div>'
        '<div class="ct-flow-label">Estimate dry bean yield</div>'
        '</div>'
        '</div>'
    )

    st.markdown(
        workflow_html,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CURRENT METHODS
    # --------------------------------------------------------

    st.markdown(
        '<div class="ct-section-title">Current Prototype Methods</div>',
        unsafe_allow_html=True,
    )

    method_1, method_2, method_3 = st.columns(
        3,
        gap="medium",
    )

    with method_1:
        st.markdown(
            """
            <div class="ct-method-card">
                <div class="ct-method-label">Pod Detection</div>
                <div class="ct-method-value">
                    Edge Impulse YOLO-Pro<br>
                    + tiled inference
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with method_2:
        st.markdown(
            """
            <div class="ct-method-card">
                <div class="ct-method-label">Maturity Estimation</div>
                <div class="ct-method-value">
                    Refined pod region<br>
                    + HSV / Lab colour analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with method_3:
        st.markdown(
            """
            <div class="ct-method-card">
                <div class="ct-method-label">Yield Estimation</div>
                <div class="ct-method-value">
                    Detected ripe pods ÷<br>
                    25 pods kg⁻¹ dry beans
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # SCOPE
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="ct-scope">
            <strong>Prototype scope.</strong>
            CocoaTrack is an experimental image-based system.
            Maturity is estimated from colour-derived features,
            and the yield output represents the dry bean yield
            associated with detected ripe pods. It should not be
            interpreted as whole-tree yield unless the image
            represents the relevant harvestable pod population
            of the entire tree.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# Unified CocoaTrack analysis workflow.
# "controlled" is retained only as an internal compatibility key.
analysis_mode = "controlled"

if analysis_mode == "controlled":
    st.markdown(
        """
        <div class="analysis-path">
            Analysis: Upload → Check image → Detect cardboard →
            Build masks → ML pod detection → Classify colour →
            Estimate maturity
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="analysis-path">
            Raw Image Mode: Upload → Isolate One Pod → Build Colour Masks →
            Detect Candidates → Select Pods → View Results
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Raw Image Mode is separate from the White-Cardboard Workflow. "
        "It uses a manual rectangle and GrabCut to isolate one pod before "
        "candidate detection."
    )


# ============================================================
# GROUPED ANALYSIS JOURNEY
# ============================================================

LOCAL_GALLERY_DIR = Path(__file__).resolve().parent / "gallery"

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

JOURNEY_STEPS = [
    ("Choose", "🖼️"),
    ("Detect", "🔎"),
    ("Results", "📊"),
]

journey_key = f"analysis_journey_step_{analysis_mode}"

if journey_key not in st.session_state:
    st.session_state[journey_key] = 0

journey_step = min(
    int(st.session_state[journey_key]),
    len(JOURNEY_STEPS) - 1,
)
st.session_state[journey_key] = journey_step


def set_journey_step(step_number: int) -> None:
    st.session_state[journey_key] = int(
        np.clip(
            step_number,
            0,
            len(JOURNEY_STEPS) - 1,
        )
    )
    st.rerun()


def journey_navigation(
    *,
    previous_step: int | None,
    next_step: int | None,
    next_label: str = "Continue",
    next_disabled: bool = False,
) -> None:
    st.markdown(
        "<div style='height:0.8rem'></div>",
        unsafe_allow_html=True,
    )

    navigation_left, navigation_right = st.columns(2)

    with navigation_left:
        if previous_step is not None:
            if st.button(
                "← Back",
                key=f"journey_back_{journey_step}",
                use_container_width=True,
            ):
                set_journey_step(previous_step)

    with navigation_right:
        if next_step is not None:
            if st.button(
                next_label,
                key=f"journey_next_{journey_step}",
                type="primary",
                use_container_width=True,
                disabled=next_disabled,
            ):
                set_journey_step(next_step)


# A clear, playful journey indicator.
journey_columns = st.columns(len(JOURNEY_STEPS))

for journey_index, (journey_name, journey_icon) in enumerate(
    JOURNEY_STEPS
):
    is_current = journey_index == journey_step
    is_complete = journey_index < journey_step

    background = (
        "#513525"
        if is_current
        else "#e8efe4"
        if is_complete
        else "#ffffff"
    )

    text_colour = (
        "#ffffff"
        if is_current
        else "#4f654a"
        if is_complete
        else "#74685f"
    )

    border_colour = (
        "#513525"
        if is_current
        else "#d7e3d3"
        if is_complete
        else "#e4d9cf"
    )

    with journey_columns[journey_index]:
        st.markdown(
            f"""
            <div style="
                background:{background};
                color:{text_colour};
                border:1px solid {border_colour};
                border-radius:16px;
                padding:12px 8px;
                text-align:center;
                min-height:82px;
                box-shadow:0 8px 18px rgba(83,54,35,0.05);
            ">
                <div style="font-size:1.35rem;">{journey_icon}</div>
                <div style="font-weight:850;margin-top:4px;">
                    {journey_name}
                </div>
                <div style="font-size:0.72rem;margin-top:2px;">
                    {"Current" if is_current else "Complete" if is_complete else f"Step {journey_index + 1}"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div style='height:0.55rem'></div>",
    unsafe_allow_html=True,
)

image_rgb = None
image_source_name = ""
image_source_size = 0


def load_selected_controlled_image() -> tuple[np.ndarray, str, int]:
    image_source_mode = st.session_state.get(
        "controlled_image_source_mode",
        "Choose from gallery",
    )

    if image_source_mode == "Choose from gallery":
        selected_path_value = st.session_state.get(
            "selected_gallery_image"
        )

        if not selected_path_value:
            raise RuntimeError(
                "No gallery image has been selected."
            )

        selected_path = Path(selected_path_value)

        if not selected_path.exists():
            raise RuntimeError(
                "The selected gallery image is no longer available."
            )

        # Load the ACTUAL analysis image in the same
        # upright orientation used by the gallery preview.
        loaded_image = ImageOps.exif_transpose(
            Image.open(selected_path)
        ).convert("RGB")

        # These CocoaTrack field photographs are intended
        # to be analysed in portrait orientation.
        if loaded_image.width > loaded_image.height:
            loaded_image = loaded_image.rotate(
                90,
                expand=True,
            )

        loaded_rgb = np.array(
            loaded_image
        )

        return (
            loaded_rgb,
            selected_path.name,
            int(selected_path.stat().st_size),
        )

    controlled_bytes = st.session_state.get(
        "controlled_uploaded_bytes"
    )

    if controlled_bytes is None:
        raise RuntimeError(
            "No uploaded controlled image is available."
        )

    loaded_rgb = np.array(
        Image.open(
            io.BytesIO(controlled_bytes)
        ).convert("RGB")
    )

    return (
        loaded_rgb,
        st.session_state.get(
            "controlled_uploaded_name",
            "uploaded_controlled_image",
        ),
        int(
            st.session_state.get(
                "controlled_uploaded_size",
                len(controlled_bytes),
            )
        ),
    )


def load_selected_raw_image() -> tuple[np.ndarray, str, int]:
    raw_bytes = st.session_state.get(
        "raw_uploaded_bytes"
    )

    if raw_bytes is None:
        raise RuntimeError(
            "No uploaded raw image is available."
        )

    loaded_rgb = np.array(
        Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")
    )

    return (
        loaded_rgb,
        st.session_state.get(
            "raw_uploaded_name",
            "uploaded_raw_image",
        ),
        int(
            st.session_state.get(
                "raw_uploaded_size",
                len(raw_bytes),
            )
        ),
    )


@st.cache_data(show_spinner=False)
def load_gallery_thumbnail(
    image_path_str: str,
    modified_time: float,
):
    """
    Load and cache a small portrait gallery preview.

    modified_time is included in the cache key so the preview
    refreshes automatically if the source image changes.
    """
    image_path = Path(image_path_str)

    image = ImageOps.exif_transpose(
        Image.open(image_path)
    ).convert("RGB")

    if image.width > image.height:
        image = image.rotate(
            90,
            expand=True,
        )

    preview = ImageOps.fit(
        image,
        (300, 400),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )

    return preview


# ============================================================
# PAGE 1 — CHOOSE IMAGE
# ============================================================

if journey_step == 0:
    stage_heading(
        1,
        "Choose Your Cocoa Image",
        "Choose one of the evaluation images or upload your own cocoa image.",
    )

    if analysis_mode == "controlled":
        image_source_mode = st.radio(
            "Image source",
            [
                "Choose from gallery",
                "Upload your own image",
            ],
            horizontal=True,
            key="controlled_image_source_mode",
        )

        image_ready = False

        if image_source_mode == "Choose from gallery":
            if not LOCAL_GALLERY_DIR.exists():
                st.warning(
                    "The gallery folder was not found at:\n\n"
                    f"`{LOCAL_GALLERY_DIR}`"
                )
                st.stop()

            gallery_images = sorted(
                [
                    path
                    for path in LOCAL_GALLERY_DIR.iterdir()
                    if path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_IMAGE_EXTENSIONS
                ],
                key=lambda path: path.name.lower(),
            )

            if not gallery_images:
                st.warning(
                    "No supported images were found in the gallery folder."
                )
                st.stop()

            st.caption(
                "All gallery previews use the same portrait frame. "
                "The original image remains unchanged for analysis."
            )

            selected_gallery_path = st.session_state.get(
                "selected_gallery_image"
            )

            number_of_columns = 3

            for row_start in range(
                0,
                len(gallery_images),
                number_of_columns,
            ):
                gallery_columns = st.columns(
                    number_of_columns
                )

                row_images = gallery_images[
                    row_start:row_start + number_of_columns
                ]

                for column_index, image_path in enumerate(
                    row_images
                ):
                    with gallery_columns[column_index]:
                        try:
                            # Gallery preview only:
                            # 1. Respect phone EXIF orientation.
                            # 2. If pixels are still landscape, rotate
                            #    the preview into portrait orientation.
                            # 3. Fit every preview into the same
                            #    portrait frame.
                            portrait_preview = load_gallery_thumbnail(
                                str(image_path),
                                image_path.stat().st_mtime,
                            )

                            is_selected = (
                                selected_gallery_path
                                == str(image_path)
                            )

                            st.image(
                                portrait_preview,
                                caption=image_path.stem,
                                use_container_width=True,
                            )

                            if st.button(
                                (
                                    "Selected"
                                    if is_selected
                                    else "Choose Image"
                                ),
                                key=(
                                    "gallery_select_"
                                    f"{image_path.name}"
                                ),
                                type=(
                                    "primary"
                                    if is_selected
                                    else "secondary"
                                ),
                                use_container_width=True,
                                disabled=is_selected,
                            ):
                                st.session_state[
                                    "selected_gallery_image"
                                ] = str(image_path)

                                reset_classification()
                                st.toast(
                                    "Image selected",
                                    icon="🍫",
                                )
                                st.rerun()

                        except Exception:
                            st.error(
                                f"Could not preview {image_path.name}"
                            )

            selected_gallery_path = st.session_state.get(
                "selected_gallery_image"
            )

            image_ready = bool(
                selected_gallery_path
                and Path(selected_gallery_path).exists()
            )

            if image_ready:
                st.success(
                    "Your image is ready for analysis."
                )
            else:
                st.info(
                    "Choose one image to continue."
                )

        else:
            uploaded_file = st.file_uploader(
                "Upload your own cocoa image",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "tif",
                    "tiff",
                ],
                key="upload_controlled",
            )

            if uploaded_file is not None:
                controlled_bytes = uploaded_file.getvalue()

                st.session_state[
                    "controlled_uploaded_bytes"
                ] = controlled_bytes

                st.session_state[
                    "controlled_uploaded_name"
                ] = uploaded_file.name

                st.session_state[
                    "controlled_uploaded_size"
                ] = len(controlled_bytes)

            image_ready = (
                "controlled_uploaded_bytes"
                in st.session_state
            )

            if image_ready:
                uploaded_preview = Image.open(
                    io.BytesIO(
                        st.session_state[
                            "controlled_uploaded_bytes"
                        ]
                    )
                ).convert("RGB")

                portrait_preview = ImageOps.fit(
                    uploaded_preview,
                    (420, 560),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )

                st.image(
                    portrait_preview,
                    caption=st.session_state.get(
                        "controlled_uploaded_name",
                        "Uploaded image",
                    ),
                    width=320,
                )

    else:
        uploaded_file = st.file_uploader(
            "Upload a raw or online cocoa image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff",
            ],
            key="upload_raw",
        )

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()

            st.session_state[
                "raw_uploaded_bytes"
            ] = raw_bytes

            st.session_state[
                "raw_uploaded_name"
            ] = uploaded_file.name

            previous_raw_signature = st.session_state.get(
                "last_raw_upload_signature"
            )

            current_raw_signature = (
                uploaded_file.name,
                len(raw_bytes),
            )

            st.session_state[
                "raw_uploaded_size"
            ] = len(raw_bytes)

            if previous_raw_signature != current_raw_signature:
                st.session_state[
                    "last_raw_upload_signature"
                ] = current_raw_signature

                for state_key in [
                    "raw_grabcut_result",
                    "selected_pod_ids",
                    "locked_selected_regions",
                    "locked_selected_pod_ids",
                    "locked_prepared_rgb",
                    "locked_green_yellow_mask",
                    "locked_red_brown_mask",
                    "locked_selection_overlay",
                ]:
                    st.session_state.pop(
                        state_key,
                        None,
                    )

        image_ready = (
            "raw_uploaded_bytes"
            in st.session_state
        )

        if image_ready:
            uploaded_preview = Image.open(
                io.BytesIO(
                    st.session_state[
                        "raw_uploaded_bytes"
                    ]
                )
            ).convert("RGB")

            portrait_preview = ImageOps.fit(
                uploaded_preview,
                (420, 560),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )

            st.image(
                portrait_preview,
                caption=st.session_state.get(
                    "raw_uploaded_name",
                    "Uploaded raw image",
                ),
                width=320,
            )

    journey_navigation(
        previous_step=None,
        next_step=1,
        next_label="Detect Cocoa Pods →",
        next_disabled=not image_ready,
    )

    st.stop()


# Load image for every later page.
try:
    if analysis_mode == "controlled":
        (
            image_rgb,
            image_source_name,
            image_source_size,
        ) = load_selected_controlled_image()
    else:
        (
            image_rgb,
            image_source_name,
            image_source_size,
        ) = load_selected_raw_image()

except Exception as error:
    st.warning(str(error))
    set_journey_step(0)


# ============================================================
# PAGE 2 — QUALITY + PREPARATION + COLOUR MASKS
# ============================================================

stage1 = check_input_quality(image_rgb)

if analysis_mode == "controlled":
    full_image_mask = np.full(
        image_rgb.shape[:2],
        255,
        dtype=np.uint8,
    )

    image_height, image_width = image_rgb.shape[:2]

    stage2 = {
        "resized_rgb": image_rgb,
        "paper_pixel_mask": full_image_mask,
        "cleaned_paper_mask": full_image_mask,
        "paper_roi_mask": full_image_mask,
        "prepared_rgb": image_rgb,
        "paper_overlay": image_rgb,
        "crop_box": (
            0,
            0,
            image_width,
            image_height,
        ),
        "scale": 1.0,
    }

else:
    raw_resized_rgb, raw_scale = resize_long_side(
        image_rgb,
        target_long_side=900,
    )

    raw_height, raw_width = raw_resized_rgb.shape[:2]

    raw_image_signature = (
        image_source_name,
        image_source_size,
        raw_width,
        raw_height,
    )

    if (
        st.session_state.get("raw_image_signature")
        != raw_image_signature
    ):
        st.session_state.raw_image_signature = (
            raw_image_signature
        )
        st.session_state.pop(
            "raw_grabcut_result",
            None,
        )
        for state_key in [
            "selected_pod_ids",
            "locked_selected_regions",
            "locked_selected_pod_ids",
            "locked_prepared_rgb",
            "locked_green_yellow_mask",
            "locked_red_brown_mask",
            "locked_selection_overlay",
        ]:
            st.session_state.pop(
                state_key,
                None,
            )

    default_width = max(
        20,
        int(raw_width * 0.70),
    )

    default_height = max(
        20,
        int(raw_height * 0.70),
    )

    centred_x = max(
        0,
        int((raw_width - default_width) / 2),
    )

    centred_y = max(
        0,
        int((raw_height - default_height) / 2),
    )

    if journey_step == 1:
        stage_heading(
            2,
            "Prepare the Raw Image",
            "Check the image, isolate one pod, and review its colour masks.",
        )

        with st.expander(
            "1. Image Quality",
            expanded=True,
        ):
            quality_image_column, quality_summary_column = st.columns(
                [2, 1]
            )

            with quality_image_column:
                st.image(
                    image_rgb,
                    caption=f"Selected Image — {image_source_name}",
                    use_container_width=True,
                )

            with quality_summary_column:
                st.metric(
                    "Quality Status",
                    stage1["status"],
                )

                st.dataframe(
                    [
                        {
                            "Measurement": key,
                            "Value": value,
                        }
                        for key, value
                        in stage1["metrics"].items()
                    ],
                    hide_index=True,
                    use_container_width=True,
                )

        with st.expander(
            "2. Isolate One Pod",
            expanded=True,
        ):
            with st.form(
                "raw_grabcut_form",
                clear_on_submit=False,
            ):
                crop_column_1, crop_column_2 = st.columns(
                    2
                )

                with crop_column_1:
                    horizontal_limit = max(
                        1,
                        int(raw_width * 0.45),
                    )

                    vertical_limit = max(
                        1,
                        int(raw_height * 0.45),
                    )

                    horizontal_offset = st.slider(
                        "Move rectangle left ↔ right",
                        min_value=-horizontal_limit,
                        max_value=horizontal_limit,
                        value=int(
                            st.session_state.get(
                                "saved_horizontal_offset",
                                0,
                            )
                        ),
                        step=1,
                    )

                    vertical_offset = st.slider(
                        "Move rectangle up ↕ down",
                        min_value=-vertical_limit,
                        max_value=vertical_limit,
                        value=int(
                            st.session_state.get(
                                "saved_vertical_offset",
                                0,
                            )
                        ),
                        step=1,
                    )

                with crop_column_2:
                    raw_box_width = st.slider(
                        "Rectangle width",
                        min_value=20,
                        max_value=max(
                            raw_width,
                            20,
                        ),
                        value=min(
                            int(
                                st.session_state.get(
                                    "saved_raw_width",
                                    default_width,
                                )
                            ),
                            max(raw_width, 20),
                        ),
                        step=1,
                    )

                    raw_box_height = st.slider(
                        "Rectangle height",
                        min_value=20,
                        max_value=max(
                            raw_height,
                            20,
                        ),
                        value=min(
                            int(
                                st.session_state.get(
                                    "saved_raw_height",
                                    default_height,
                                )
                            ),
                            max(raw_height, 20),
                        ),
                        step=1,
                    )

                raw_x = int(
                    np.clip(
                        centred_x + horizontal_offset,
                        0,
                        max(
                            raw_width - raw_box_width,
                            0,
                        ),
                    )
                )

                raw_y = int(
                    np.clip(
                        centred_y + vertical_offset,
                        0,
                        max(
                            raw_height - raw_box_height,
                            0,
                        ),
                    )
                )

                grabcut_iterations = st.slider(
                    "GrabCut refinement iterations",
                    min_value=1,
                    max_value=8,
                    value=int(
                        st.session_state.get(
                            "saved_grabcut_iterations",
                            3,
                        )
                    ),
                    step=1,
                )

                run_grabcut = st.form_submit_button(
                    "Isolate the selected pod",
                    type="primary",
                    use_container_width=True,
                )

            if run_grabcut:
                st.session_state.saved_horizontal_offset = int(
                    horizontal_offset
                )
                st.session_state.saved_vertical_offset = int(
                    vertical_offset
                )
                st.session_state.saved_raw_x = int(
                    raw_x
                )
                st.session_state.saved_raw_y = int(
                    raw_y
                )
                st.session_state.saved_raw_width = int(
                    raw_box_width
                )
                st.session_state.saved_raw_height = int(
                    raw_box_height
                )
                st.session_state.saved_grabcut_iterations = int(
                    grabcut_iterations
                )

                with st.spinner(
                    "Removing the background..."
                ):
                    st.session_state.raw_grabcut_result = (
                        isolate_raw_pod_with_grabcut(
                            raw_resized_rgb,
                            x=raw_x,
                            y=raw_y,
                            width=raw_box_width,
                            height=raw_box_height,
                            iterations=grabcut_iterations,
                        )
                    )

                st.session_state.pop(
                    "selected_pod_ids",
                    None,
                )

                st.toast(
                    "Pod isolated",
                    icon="✨",
                )
                st.rerun()

            raw_result = st.session_state.get(
                "raw_grabcut_result"
            )

            if raw_result is None:
                preview = raw_resized_rgb.copy()

                preview_x = int(
                    st.session_state.get(
                        "saved_raw_x",
                        centred_x,
                    )
                )

                preview_y = int(
                    st.session_state.get(
                        "saved_raw_y",
                        centred_y,
                    )
                )

                preview_width = int(
                    st.session_state.get(
                        "saved_raw_width",
                        default_width,
                    )
                )

                preview_height = int(
                    st.session_state.get(
                        "saved_raw_height",
                        default_height,
                    )
                )

                cv2.rectangle(
                    preview,
                    (preview_x, preview_y),
                    (
                        min(
                            preview_x + preview_width,
                            raw_width - 1,
                        ),
                        min(
                            preview_y + preview_height,
                            raw_height - 1,
                        ),
                    ),
                    (255, 220, 0),
                    4,
                )

                st.image(
                    preview,
                    caption="Rectangle preview",
                    use_container_width=True,
                )

                st.info(
                    "Adjust the rectangle and isolate the pod before continuing."
                )

            else:
                preview_column, mask_column, result_column = st.columns(
                    3
                )

                with preview_column:
                    st.image(
                        raw_result["preview"],
                        caption="Selected Rectangle",
                        use_container_width=True,
                    )

                with mask_column:
                    st.image(
                        raw_result["crop_mask"],
                        caption="Foreground Mask",
                        clamp=True,
                        use_container_width=True,
                    )

                with result_column:
                    st.image(
                        raw_result["isolated_rgb"],
                        caption="Isolated Pod",
                        use_container_width=True,
                    )

                if raw_result.get(
                    "used_fallback",
                    False,
                ):
                    st.warning(
                        "GrabCut used the selected rectangle as a fallback. "
                        "A tighter rectangle may improve the result."
                    )

        raw_result = st.session_state.get(
            "raw_grabcut_result"
        )

        if raw_result is not None:
            stage2 = {
                "resized_rgb": raw_resized_rgb,
                "paper_pixel_mask": raw_result[
                    "crop_mask"
                ],
                "cleaned_paper_mask": raw_result[
                    "crop_mask"
                ],
                "paper_roi_mask": raw_result[
                    "crop_mask"
                ],
                "prepared_rgb": raw_result[
                    "isolated_rgb"
                ],
                "paper_overlay": raw_result[
                    "preview"
                ],
                "crop_box": raw_result[
                    "crop_box"
                ],
                "scale": raw_scale,
            }

            stage3 = prepare_colour_masks(
                stage2["prepared_rgb"],
                stage2["paper_roi_mask"],
            )

            with st.expander(
                "3. Colour Masks",
                expanded=True,
            ):
                mask_column_1, mask_column_2, mask_column_3 = st.columns(
                    3
                )

                with mask_column_1:
                    st.image(
                        stage3["green_yellow_mask"],
                        caption="Green-Yellow",
                        clamp=True,
                        use_container_width=True,
                    )

                with mask_column_2:
                    st.image(
                        stage3["red_brown_mask"],
                        caption="Red-Brown",
                        clamp=True,
                        use_container_width=True,
                    )

                with mask_column_3:
                    st.image(
                        stage3["combined_rgb"],
                        caption="Combined Mask",
                        use_container_width=True,
                    )

        journey_navigation(
            previous_step=0,
            next_step=2,
            next_label="Find Cocoa Pods →",
            next_disabled=raw_result is None,
        )

        st.stop()

    raw_result = st.session_state.get(
        "raw_grabcut_result"
    )

    if raw_result is None:
        set_journey_step(1)

    stage2 = {
        "resized_rgb": raw_resized_rgb,
        "paper_pixel_mask": raw_result[
            "crop_mask"
        ],
        "cleaned_paper_mask": raw_result[
            "crop_mask"
        ],
        "paper_roi_mask": raw_result[
            "crop_mask"
        ],
        "prepared_rgb": raw_result[
            "isolated_rgb"
        ],
        "paper_overlay": raw_result[
            "preview"
        ],
        "crop_box": raw_result[
            "crop_box"
        ],
        "scale": raw_scale,
    }


stage3 = prepare_colour_masks(
    stage2["prepared_rgb"],
    stage2["paper_roi_mask"],
)



# ============================================================
# PAGE 3 — ML POD DETECTION
# ============================================================

# ============================================================
# OPTIONAL ML FOCUS CROP
# ============================================================

# Default: detector receives the full original image.
analysis_rgb = image_rgb.copy()

crop_mode = st.session_state.get(
    "ml_detection_area_mode",
    "Full image",
)

horizontal_crop = st.session_state.get(
    "ml_crop_horizontal",
    (0, 100),
)

vertical_crop = st.session_state.get(
    "ml_crop_vertical",
    (0, 100),
)

# ============================================================
# DETECTION AREA UI
# Image preview on LEFT, controls on RIGHT
# ============================================================

if journey_step == 1:
    preview_column, control_column = st.columns(
        [1.15, 1.0],
        gap="large",
    )

    # --------------------------------------------------------
    # RIGHT SIDE — FULL IMAGE / CROP CONTROLS
    # --------------------------------------------------------
    with control_column:
        st.markdown("### Detection Area")

        crop_mode = st.radio(
            "Choose the area used for ML detection",
            [
                "Full image",
                "Crop / focus area",
            ],
            horizontal=False,
            key="ml_detection_area_mode",
        )

        if crop_mode == "Crop / focus area":
            st.markdown("#### Focus Area")

            horizontal_crop = st.slider(
                "Horizontal area (%)",
                min_value=0,
                max_value=100,
                value=tuple(horizontal_crop),
                key="ml_crop_horizontal",
            )

            vertical_crop = st.slider(
                "Vertical area (%)",
                min_value=0,
                max_value=100,
                value=tuple(vertical_crop),
                key="ml_crop_vertical",
            )

            st.caption(
                "Adjust the sliders to limit where the ML "
                "detector searches for cocoa pods."
            )

        else:
            st.caption(
                "The full image will be used for ML pod detection."
            )


# Re-read widget values after Streamlit updates.
crop_mode = st.session_state.get(
    "ml_detection_area_mode",
    "Full image",
)

horizontal_crop = st.session_state.get(
    "ml_crop_horizontal",
    (0, 100),
)

vertical_crop = st.session_state.get(
    "ml_crop_vertical",
    (0, 100),
)


# ============================================================
# BUILD IMAGE THAT WILL ACTUALLY GO TO YOLO
# ============================================================

analysis_rgb = image_rgb.copy()

if crop_mode == "Crop / focus area":
    image_h, image_w = image_rgb.shape[:2]

    x1_percent, x2_percent = horizontal_crop
    y1_percent, y2_percent = vertical_crop

    if (
        x2_percent - x1_percent >= 5
        and y2_percent - y1_percent >= 5
    ):
        x1 = int(
            image_w * x1_percent / 100
        )
        x2 = int(
            image_w * x2_percent / 100
        )

        y1 = int(
            image_h * y1_percent / 100
        )
        y2 = int(
            image_h * y2_percent / 100
        )

        analysis_rgb = image_rgb[
            y1:y2,
            x1:x2,
        ].copy()

    elif journey_step == 1:
        with control_column:
            st.warning(
                "The selected crop is too small. "
                "Increase the focus area."
            )


# ------------------------------------------------------------
# LEFT SIDE — SMALLER IMAGE PREVIEW
# ------------------------------------------------------------
if journey_step == 1:
    with preview_column:
        st.markdown("### Detection Preview")

        st.image(
            analysis_rgb,
            caption=(
                "Selected focus area"
                if crop_mode == "Crop / focus area"
                else "Full image"
            ),
            width="stretch",
        )

        st.caption(
            "This is the exact image sent into "
            "the tiled YOLO-Pro detector."
        )


# Cache ML detection for the exact image/crop so ordinary
# Streamlit reruns do not repeatedly relaunch Edge Impulse.
import hashlib

_analysis_array = np.ascontiguousarray(analysis_rgb)
_analysis_key = hashlib.sha256(
    _analysis_array.tobytes()
).hexdigest()

if (
    st.session_state.get("_cocoatrack_detection_key")
    != _analysis_key
    or "_cocoatrack_stage4" not in st.session_state
):
    print(
        "COCOATRACK_DIAG: NEW IMAGE/CROP - RUNNING DETECTOR",
        flush=True,
    )

    st.session_state["_cocoatrack_stage4"] = (
        detect_pods_with_edge_impulse(
            prepared_rgb=analysis_rgb,
            combined_mask=None,
        )
    )

    st.session_state["_cocoatrack_detection_key"] = (
        _analysis_key
    )
else:
    print(
        "COCOATRACK_DIAG: SAME IMAGE/CROP - USING CACHED RESULT",
        flush=True,
    )

stage4 = st.session_state["_cocoatrack_stage4"]

if journey_step == 1:
    stage_heading(
        3,
        "ML Cocoa Pod Detection",
        "The trained Edge Impulse YOLO-Pro model automatically detects cocoa pods.",
    )

    detection_image_column, detection_summary_column = st.columns(
        [2, 1]
    )

    with detection_image_column:
        ml_left, ml_image, ml_right = st.columns(
            [0.7, 2.6, 0.7]
        )

        with ml_image:
            st.image(
                stage4["output_image"],
                caption="ML-Detected Cocoa Pods",
                width="stretch",
            )

    with detection_summary_column:
        st.metric(
            "Detected Pods",
            len(stage4["accepted_regions"]),
        )

        if "inference_seconds" in stage4:
            st.metric(
                "Inference Time",
                f'{stage4["inference_seconds"]:.2f} s',
            )

        if len(stage4["accepted_regions"]) > 0:
            st.success(
                "Detection complete. Continue to review the detected pods."
            )
        else:
            st.warning(
                "No cocoa pods were detected in this image."
            )

    journey_navigation(
        previous_step=0,
        next_step=2,
        next_label="Analyze Results →",
        next_disabled=(
            len(stage4["accepted_regions"]) == 0
        ),
    )

    st.stop()


# ============================================================
# AUTOMATIC ML DETECTION OUTPUT
# ============================================================

# Every pod accepted by the trained ML detector automatically
# continues into colour, maturity, and yield analysis.
selected_regions = stage4["accepted_regions"]

if not selected_regions:
    st.warning(
        "No cocoa pods were detected in this image."
    )
    st.stop()



def segment_ml_pods_for_maturity(
    image_rgb: np.ndarray,
    selected_regions: list,
) -> dict:
    """
    Refine each accepted ML bounding box into a pod-shaped mask.

    YOLO remains responsible for finding the pod.

    Inside each accepted YOLO box:
        1. GrabCut estimates foreground/background.
        2. Morphology cleans the foreground.
        3. Watershed separates foreground structures.
        4. The most plausible central pod component is retained.
        5. A slight erosion removes uncertain edge pixels.

    If segmentation fails, an ellipse inside the detected box
    is used rather than treating the whole bounding box as pod.
    """

    image_h, image_w = image_rgb.shape[:2]

    label_image = np.zeros(
        (image_h, image_w),
        dtype=np.int32,
    )

    overlay = image_rgb.copy()

    pods = []

    for pod_id, region in enumerate(
        selected_regions,
        start=1,
    ):
        x, y, w, h = region["bbox"]

        x = max(0, int(x))
        y = max(0, int(y))
        w = max(1, int(w))
        h = max(1, int(h))

        w = min(
            w,
            image_w - x,
        )

        h = min(
            h,
            image_h - y,
        )

        if w < 10 or h < 10:
            continue

        crop = image_rgb[
            y:y + h,
            x:x + w
        ].copy()

        ch, cw = crop.shape[:2]

        # ====================================================
        # GRABCUT INITIALISATION
        #
        # Do NOT simply initialise the whole rectangle as
        # foreground. The box border is strong background,
        # the central part is probable/definite foreground.
        # ====================================================

        gc_mask = np.full(
            (ch, cw),
            cv2.GC_PR_BGD,
            dtype=np.uint8,
        )

        border_x = max(
            2,
            int(cw * 0.06),
        )

        border_y = max(
            2,
            int(ch * 0.06),
        )

        # Definite background around the boundary.
        gc_mask[
            :border_y,
            :
        ] = cv2.GC_BGD

        gc_mask[
            ch - border_y:,
            :
        ] = cv2.GC_BGD

        gc_mask[
            :,
            :border_x
        ] = cv2.GC_BGD

        gc_mask[
            :,
            cw - border_x:
        ] = cv2.GC_BGD

        centre = (
            cw // 2,
            ch // 2,
        )

        # Probable foreground ellipse.
        probable_axes = (
            max(2, int(cw * 0.42)),
            max(2, int(ch * 0.42)),
        )

        cv2.ellipse(
            gc_mask,
            centre,
            probable_axes,
            0,
            0,
            360,
            cv2.GC_PR_FGD,
            -1,
        )

        # Smaller definite-foreground core.
        definite_axes = (
            max(2, int(cw * 0.20)),
            max(2, int(ch * 0.20)),
        )

        cv2.ellipse(
            gc_mask,
            centre,
            definite_axes,
            0,
            0,
            360,
            cv2.GC_FGD,
            -1,
        )

        bg_model = np.zeros(
            (1, 65),
            np.float64,
        )

        fg_model = np.zeros(
            (1, 65),
            np.float64,
        )

        method = "GrabCut + Watershed"

        try:
            cv2.grabCut(
                crop,
                gc_mask,
                None,
                bg_model,
                fg_model,
                5,
                cv2.GC_INIT_WITH_MASK,
            )

            foreground = np.where(
                (
                    gc_mask == cv2.GC_FGD
                )
                |
                (
                    gc_mask == cv2.GC_PR_FGD
                ),
                255,
                0,
            ).astype(np.uint8)

        except Exception:
            foreground = np.zeros(
                (ch, cw),
                dtype=np.uint8,
            )

        # ====================================================
        # MORPHOLOGICAL CLEANING
        # ====================================================

        k = max(
            3,
            int(
                min(cw, ch)
                * 0.025
            ),
        )

        if k % 2 == 0:
            k += 1

        k = min(k, 9)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            kernel,
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # ====================================================
        # WATERSHED
        # ====================================================

        candidate = foreground.copy()

        try:
            distance = cv2.distanceTransform(
                foreground,
                cv2.DIST_L2,
                5,
            )

            max_distance = float(
                distance.max()
            )

            if max_distance > 0:
                _, sure_fg = cv2.threshold(
                    distance,
                    0.38 * max_distance,
                    255,
                    0,
                )

                sure_fg = sure_fg.astype(
                    np.uint8
                )

                sure_bg = cv2.dilate(
                    foreground,
                    kernel,
                    iterations=2,
                )

                unknown = cv2.subtract(
                    sure_bg,
                    sure_fg,
                )

                _, markers = cv2.connectedComponents(
                    sure_fg
                )

                markers = (
                    markers.astype(np.int32)
                    + 1
                )

                markers[
                    unknown == 255
                ] = 0

                crop_bgr = cv2.cvtColor(
                    crop,
                    cv2.COLOR_RGB2BGR,
                )

                watershed = cv2.watershed(
                    crop_bgr,
                    markers,
                )

                candidate = np.where(
                    watershed > 1,
                    255,
                    0,
                ).astype(np.uint8)

                # Watershed cannot extend outside the
                # GrabCut foreground.
                candidate = cv2.bitwise_and(
                    candidate,
                    foreground,
                )

        except Exception:
            candidate = foreground.copy()
            method = "GrabCut"

        # ====================================================
        # KEEP CENTRAL COMPONENT; OTHERWISE LARGEST COMPONENT
        # ====================================================

        binary = (
            candidate > 0
        ).astype(np.uint8)

        num_labels, component_labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        chosen = None

        centre_label = int(
            component_labels[
                ch // 2,
                cw // 2,
            ]
        )

        if centre_label > 0:
            chosen = centre_label

        elif num_labels > 1:
            areas = stats[
                1:,
                cv2.CC_STAT_AREA,
            ]

            chosen = (
                int(
                    np.argmax(areas)
                )
                + 1
            )

        if chosen is not None:
            local_mask = np.where(
                component_labels == chosen,
                255,
                0,
            ).astype(np.uint8)

        else:
            local_mask = np.zeros(
                (ch, cw),
                dtype=np.uint8,
            )

        # ====================================================
        # VALIDITY CHECK
        # ====================================================

        bbox_area = max(
            cw * ch,
            1,
        )

        segmented_area = int(
            np.count_nonzero(
                local_mask
            )
        )

        fraction = (
            segmented_area
            / bbox_area
        )

        # A result covering almost the entire rectangle is
        # not useful segmentation. A tiny result is also bad.
        segmentation_failed = (
            fraction < 0.08
            or fraction > 0.92
        )

        if segmentation_failed:
            # Conservative non-rectangular fallback.
            local_mask = np.zeros(
                (ch, cw),
                dtype=np.uint8,
            )

            cv2.ellipse(
                local_mask,
                (
                    cw // 2,
                    ch // 2,
                ),
                (
                    max(
                        2,
                        int(cw * 0.42),
                    ),
                    max(
                        2,
                        int(ch * 0.44),
                    ),
                ),
                0,
                0,
                360,
                255,
                -1,
            )

            method = "Ellipse fallback"

        # ====================================================
        # SLIGHT INWARD TRIM
        #
        # Boundary pixels are the most likely locations for
        # leaf/background contamination.
        # ====================================================

        erosion_size = max(
            3,
            int(
                min(cw, ch)
                * 0.018
            ),
        )

        if erosion_size % 2 == 0:
            erosion_size += 1

        erosion_size = min(
            erosion_size,
            7,
        )

        erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                erosion_size,
                erosion_size,
            ),
        )

        eroded = cv2.erode(
            local_mask,
            erosion_kernel,
            iterations=1,
        )

        if (
            np.count_nonzero(eroded)
            > 0.55
            * np.count_nonzero(
                local_mask
            )
        ):
            local_mask = eroded

        # ====================================================
        # CONTOUR
        # ====================================================

        contours, _ = cv2.findContours(
            local_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        local_contour = max(
            contours,
            key=cv2.contourArea,
        )

        full_contour = (
            local_contour.copy()
        )

        full_contour[
            :, :, 0
        ] += x

        full_contour[
            :, :, 1
        ] += y

        # ====================================================
        # WRITE MASK INTO FULL-SIZE LABEL IMAGE
        # ====================================================

        target = label_image[
            y:y + ch,
            x:x + cw
        ]

        assign = (
            (local_mask > 0)
            & (target == 0)
        )

        target[
            assign
        ] = int(pod_id)

        label_image[
            y:y + ch,
            x:x + cw
        ] = target

        # Draw actual contour, not rectangle.
        cv2.drawContours(
            overlay,
            [full_contour],
            -1,
            (0, 255, 0),
            3,
        )

        bx, by, bw, bh = cv2.boundingRect(
            full_contour
        )

        # Large readable pod number label.
        label = str(pod_id)

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            5,
        )

        label_x = max(0, bx)
        label_y = max(
            th + 10,
            by,
        )

        # White label background.
        cv2.rectangle(
            overlay,
            (
                label_x,
                label_y - th - 10,
            ),
            (
                label_x + tw + 14,
                label_y + baseline + 4,
            ),
            (255, 255, 255),
            -1,
        )

        # Thin dark border.
        cv2.rectangle(
            overlay,
            (
                label_x,
                label_y - th - 10,
            ),
            (
                label_x + tw + 14,
                label_y + baseline + 4,
            ),
            (45, 35, 30),
            2,
        )

        cv2.putText(
            overlay,
            label,
            (
                label_x + 10,
                label_y - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (35, 30, 27),
            5,
        )

        pods.append(
            {
                "Pod_ID": int(pod_id),
                "bbox": (
                    bx,
                    by,
                    bw,
                    bh,
                ),
                "contour": full_contour,
                "Segmentation_Method": method,
                "Segmented_Area": int(
                    np.count_nonzero(
                        local_mask
                    )
                ),
            }
        )

    return {
        "label_image": label_image,
        "overlay_ids": overlay,
        "pods": pods,
    }


# ============================================================
# PAGE 4 — COMBINED RESULTS
# ============================================================

print(
    f"COCOATRACK_DIAG: STAGE5 START - PODS={len(selected_regions)} "
    f"IMAGE={analysis_rgb.shape}",
    flush=True,
)

stage5 = segment_ml_pods_for_maturity(
    analysis_rgb,
    selected_regions,
)

print(
    f"COCOATRACK_DIAG: STAGE5 DONE - SEGMENTED={len(stage5['pods'])}",
    flush=True,
)

print("COCOATRACK_DIAG: STAGE6 START", flush=True)

stage6 = extract_features(
    analysis_rgb,
    stage5["label_image"],
    stage5["pods"],
)

print(
    f"COCOATRACK_DIAG: STAGE6 DONE - ROWS={len(stage6)}",
    flush=True,
)

if stage6.empty:
    st.warning(
        "No features were extracted."
    )
    st.stop()

# Colour masks are calculated only AFTER ML detection,
# for maturity estimation inside detected pod regions.
full_analysis_mask = np.full(
    analysis_rgb.shape[:2],
    255,
    dtype=np.uint8,
)

final_colour_masks = prepare_colour_masks(
    analysis_rgb,
    full_analysis_mask,
)

print("COCOATRACK_DIAG: COLOUR MASKS DONE", flush=True)
print("COCOATRACK_DIAG: STAGE7 START", flush=True)

stage7 = classify_by_mask_overlap(
    features_df=stage6,
    label_image=stage5["label_image"],
    green_yellow_mask=final_colour_masks["green_yellow_mask"],
    red_brown_mask=final_colour_masks["red_brown_mask"],
)

print(
    f"COCOATRACK_DIAG: STAGE7 DONE - ROWS={len(stage7)}",
    flush=True,
)

print("COCOATRACK_DIAG: STAGE8 START", flush=True)

stage8 = classify_maturity_from_hsv(
    stage7,
    uncertainty_threshold=0.46,
)

print(
    f"COCOATRACK_DIAG: STAGE8 DONE - ROWS={len(stage8)}",
    flush=True,
)

# ============================================================
# SMALL-POD MATURITY SAFEGUARD
#
# Colour remains the main maturity indicator.
# A pod predicted as RIPE is changed to UNCERTAIN only if its
# detected area is unusually small relative to the other pods
# in the same image.
#
# This avoids assuming that "brown = ripe" for very small pods.
# ============================================================

if (
    not stage8.empty
    and "Area" in stage8.columns
):
    median_pod_area = float(
        stage8["Area"].median()
    )

    # Relative size of each pod compared with the median pod
    # in this particular image.
    stage8["Relative_Pod_Size"] = (
        stage8["Area"]
        / max(median_pod_area, 1.0)
    )

    SMALL_POD_RATIO = 0.40

    small_ripe_mask = (
        (stage8["Maturity_Class"] == "RIPE")
        & (
            stage8["Relative_Pod_Size"]
            < SMALL_POD_RATIO
        )
    )

    stage8.loc[
        small_ripe_mask,
        "Maturity_Class"
    ] = "UNCERTAIN"

    stage8["Small_Pod_Safeguard"] = False

    stage8.loc[
        small_ripe_mask,
        "Small_Pod_Safeguard"
    ] = True

# Simple Pod Number -> Maturity table
pod_summary_df = stage8[
    ["Pod_ID", "Maturity_Class"]
].copy()

pod_summary_df["Pod_ID"] = (
    pod_summary_df["Pod_ID"]
    .astype(int)
    .map(lambda pod_id: f"Pod {pod_id}")
)

pod_summary_df.columns = [
    "Pod Number",
    "Maturity",
]

# ============================================================
# COCOATRACK YIELD ESTIMATION
#
# Literature-supported prototype:
# estimated dry bean yield (kg) = ripe pod count / 25
#
# Only pods estimated as RIPE contribute to the current
# harvestable-yield estimate.
# ============================================================

estimated_ripe_pods = int(
    stage8["Maturity_Class"]
    .astype(str)
    .str.strip()
    .str.upper()
    .eq("RIPE")
    .sum()
)

POD_INDEX = 25.0  # pods per kg dry cocoa beans

estimated_dry_yield_kg = (
    estimated_ripe_pods / POD_INDEX
)



# ============================================================
# COCOATRACK HARVESTABLE YIELD ESTIMATION
#
# Prototype equation:
# Dry bean yield (kg) = estimated ripe pod count / 25
# ============================================================

estimated_ripe_pods = int(
    (
        stage8["Maturity_Class"]
        .astype(str)
        .str.upper()
        == "RIPE"
    ).sum()
)

POD_INDEX = 25.0

estimated_dry_yield_kg = (
    estimated_ripe_pods / POD_INDEX
)


maturity_majority = stage8.attrs.get(
    "maturity_majority",
    "UNCERTAIN",
)

image_majority = stage7.attrs.get(
    "image_majority",
    "UNCERTAIN",
)

green_candidate_count = stage7.attrs.get(
    "green_candidate_count",
    0,
)

red_candidate_count = stage7.attrs.get(
    "red_candidate_count",
    0,
)

unripe_count = int(
    (
        stage8["Maturity_Class"]
        == "UNRIPE"
    ).sum()
)

half_ripe_count = int(
    (
        stage8["Maturity_Class"]
        == "HALF-RIPE"
    ).sum()
)

ripe_count = int(
    (
        stage8["Maturity_Class"]
        == "RIPE"
    ).sum()
)

final_overlay = stage5["overlay_ids"].copy()


stage_heading(
    3,
    "Your Cocoa Pod Results",
    "Detected pod regions are refined within each ML detection before colour and maturity analysis.",
)

st.success(
    "ML detections retained; pod regions refined for colour and maturity analysis."
)

st.markdown(
    f"""
    <div style="
        background:linear-gradient(135deg,#513525,#7a523a);
        color:white;
        border-radius:22px;
        padding:20px;
        margin-bottom:16px;
        box-shadow:0 16px 34px rgba(83,54,35,0.16);
    ">
        <div style="font-size:0.85rem;opacity:0.85;">
            ANALYSIS COMPLETE
        </div>
        <div style="font-size:1.55rem;font-weight:900;margin-top:4px;">
            {len(stage8)} cocoa pod{"s" if len(stage8) != 1 else ""} analyzed
        </div>
        <div style="margin-top:6px;opacity:0.9;">
            Main colour: {image_majority} · Maturity: {maturity_majority}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_metric_1, summary_metric_2, summary_metric_3, summary_metric_4 = st.columns(
    4
)

with summary_metric_1:
    st.metric(
        "Detected Pods",
        len(stage8),
    )

with summary_metric_2:
    st.metric(
        "Green-Yellow",
        green_candidate_count,
    )

with summary_metric_3:
    st.metric(
        "Red-Brown",
        red_candidate_count,
    )

with summary_metric_4:
    st.metric(
        "Maturity Majority",
        maturity_majority,
    )

result_tabs = st.tabs(
    [
        "Overview",
        "Maturity",
    ]
)

with result_tabs[0]:

    overview_left, overview_right = st.columns(
        [1.12, 1.0],
        gap="large",
    )

    # ========================================================
    # LEFT — FINAL ANNOTATED RESULT
    # ========================================================

    with overview_left:

        st.markdown("### Final Annotated Result")

        st.image(
            final_overlay,
            caption="Final CocoaTrack Result",
            width="stretch",
        )

    # ========================================================
    # RIGHT — YIELD + MATURITY
    # ========================================================

    with overview_right:

        st.markdown("### Yield Estimate")

        yield_col1, yield_col2 = st.columns(
            2,
            gap="medium",
        )

        with yield_col1:
            st.metric(
                "Ripe Pods",
                estimated_ripe_pods,
            )

        with yield_col2:
            st.metric(
                "Dry Bean Yield",
                f"{estimated_dry_yield_kg:.3f} kg",
            )

        st.caption(
            "Estimated dry bean yield from detected ripe pods "
            "using a prototype pod index of 25 pods kg⁻¹ dry beans."
        )

        st.markdown("### Estimated Pod Maturity")

        maturity_rows = []

        for _, maturity_row in stage8.iterrows():

            pod_number = int(
                maturity_row["Pod_ID"]
            )

            maturity_class = str(
                maturity_row["Maturity_Class"]
            )

            confidence = maturity_row.get(
                "Maturity_Confidence",
                None,
            )

            class_key = (
                maturity_class
                .lower()
                .replace("-", "")
                .replace(" ", "")
            )

            if maturity_class == "RIPE":
                css_class = "maturity-ripe"

            elif maturity_class == "HALF-RIPE":
                css_class = "maturity-half"

            elif maturity_class == "UNRIPE":
                css_class = "maturity-unripe"

            else:
                css_class = "maturity-uncertain"

            maturity_rows.append(
                (
                    pod_number,
                    maturity_class,
                    css_class,
                    confidence,
                )
            )

        for row_start in range(
            0,
            len(maturity_rows),
            2,
        ):
            card_columns = st.columns(
                2,
                gap="small",
            )

            row_items = maturity_rows[
                row_start:row_start + 2
            ]

            for col_idx, item in enumerate(
                row_items
            ):
                (
                    pod_number,
                    maturity_class,
                    css_class,
                    confidence,
                ) = item

                confidence_text = ""

                if (
                    confidence is not None
                    and pd.notna(confidence)
                ):
                    confidence_text = (
                        f'<div class="maturity-confidence">'
                        f'Confidence: {float(confidence) * 100:.0f}%'
                        f'</div>'
                    )

                with card_columns[col_idx]:

                    card_html = (
                        f'<div class="maturity-card">'
                        f'<div class="maturity-card-top">'
                        f'<div class="maturity-pod-number">Pod {pod_number}</div>'
                        f'<div class="maturity-status {css_class}">{maturity_class}</div>'
                        f'</div>'
                        f'<div class="maturity-main">Pod {pod_number}</div>'
                        f'{confidence_text}'
                        f'</div>'
                    )

                    st.markdown(
                        card_html,
                        unsafe_allow_html=True,
                    )

        st.caption(
            "Maturity is an experimental colour-based estimate."
        )

with result_tabs[1]:
    st.markdown("#### Maturity Estimation")

    maturity_columns = [
        "Pod_ID",
        "H_Mean",
        "S_Mean",
        "V_Mean",
        "Maturity_Confidence",
        "Maturity_Class",
        "Image_Maturity_Majority",
    ]

    st.dataframe(
        stage8[maturity_columns],
        hide_index=True,
        use_container_width=True,
    )

download_column_1, download_column_2 = st.columns(
    2
)

with download_column_1:
    st.download_button(
        "Download Analysis Table",
        data=stage8.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "cocoatrack_colour_and_"
            "maturity_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

with download_column_2:
    st.download_button(
        "Download Annotated Image",
        data=to_png_bytes(final_overlay),
        file_name=(
            "cocoatrack_final_colour_"
            "and_maturity.png"
        ),
        mime="image/png",
        use_container_width=True,
    )

result_back_column, result_restart_column = st.columns(
    2
)

with result_back_column:
    if st.button(
        "← Back to ML detection",
        use_container_width=True,
    ):
        set_journey_step(1)

with result_restart_column:
    if st.button(
        "Analyze Another Image",
        type="primary",
        use_container_width=True,
    ):
        st.session_state[journey_key] = 0

        for state_key in [
            "selected_gallery_image",
            "controlled_uploaded_bytes",
            "controlled_uploaded_name",
            "controlled_uploaded_size",
            "raw_uploaded_bytes",
            "raw_uploaded_name",
            "raw_uploaded_size",
            "raw_grabcut_result",
            "selected_pod_ids",
        ]:
            st.session_state.pop(
                state_key,
                None,
            )

        reset_classification()
        st.rerun()

st.markdown(
    """
    <div class="footer-note">
        CocoaTrack · Experimental cocoa pod analysis
    </div>
    """,
    unsafe_allow_html=True,
)
