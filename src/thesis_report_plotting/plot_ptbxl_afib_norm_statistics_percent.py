from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABEL_ORDER = ["AFIB", "NORM"]

LEAD_CANONICAL_MAP = {
    "I": "I",
    "II": "II",
    "III": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}

LEAD_TO_INDEX = {lead: i for i, lead in enumerate(LEAD_ORDER)}
LEAD_TOKEN_REGEX = re.compile(r"(?i)(AVR|AVL|AVF|V[1-6]|III|II|I)")
RANGE_REGEX = re.compile(
    r"(?i)\b(AVR|AVL|AVF|V[1-6]|III|II|I)\s*-\s*(AVR|AVL|AVF|V[1-6]|III|II|I)\b"
)


def parse_scp_codes(value: str) -> Dict[str, float]:
    if pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load PTB-XL metadata from ptbxl_database.csv and keep only AFIB/NORM-related records.
    """
    csv_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(csv_path)

    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)
    df["has_afib"] = df["scp_codes"].apply(lambda codes: "AFIB" in codes)
    df["has_norm"] = df["scp_codes"].apply(lambda codes: "NORM" in codes)

    return df[df["has_afib"] | df["has_norm"]].copy()


def build_record_counts(df: pd.DataFrame) -> pd.Series:
    """
    Count total AFIB and NORM recordings.
    """
    return pd.Series({
        "AFIB": int(df["has_afib"].sum()),
        "NORM": int(df["has_norm"].sum()),
    })


def build_unique_patient_counts(df: pd.DataFrame) -> pd.Series:
    """
    Count unique patients for AFIB and NORM.
    """
    af_patients = df.loc[df["has_afib"], "patient_id"].dropna().nunique()
    norm_patients = df.loc[df["has_norm"], "patient_id"].dropna().nunique()

    return pd.Series({
        "AFIB": int(af_patients),
        "NORM": int(norm_patients),
    })


def build_lead_presence_df(total_record_counts: pd.Series) -> pd.DataFrame:
    """
    PTB-XL records are standard 12-lead ECGs.
    Each included record contributes one observation to each lead.
    """
    return pd.DataFrame(
        {label: [int(total_record_counts[label])] * len(LEAD_ORDER) for label in LABEL_ORDER},
        index=LEAD_ORDER,
    )


def canonicalize_token(token: str) -> str:
    return LEAD_CANONICAL_MAP[token.upper()]


def expand_range(start: str, end: str) -> List[str]:
    """
    Expand compact lead ranges such as I-V1 or I-AVF.
    """
    start_canonical = canonicalize_token(start)
    end_canonical = canonicalize_token(end)

    i = LEAD_TO_INDEX[start_canonical]
    j = LEAD_TO_INDEX[end_canonical]
    lo, hi = sorted((i, j))

    return LEAD_ORDER[lo:hi + 1]


def extract_problem_leads(value: object) -> List[str]:
    """
    Extract problematic leads from the PTB-XL metadata field 'electrodes_problems'.

    Supported forms:
    - single lead: V1
    - comma-separated list: II,III,AVF
    - compact range: I-V1 or I-AVF
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    leads: List[str] = []
    seen = set()

    for start, end in RANGE_REGEX.findall(text):
        for lead in expand_range(start, end):
            if lead not in seen:
                leads.append(lead)
                seen.add(lead)

    text_without_ranges = RANGE_REGEX.sub(" ", text)
    for token in LEAD_TOKEN_REGEX.findall(text_without_ranges):
        lead = canonicalize_token(token)
        if lead not in seen:
            leads.append(lead)
            seen.add(lead)

    return leads


def build_nonworking_counts_df(df: pd.DataFrame, dead_lead_column: str = "electrodes_problems") -> pd.DataFrame:
    """
    Count non-working leads per label using the metadata field 'electrodes_problems'.
    """
    nonworking = pd.DataFrame(0, index=LEAD_ORDER, columns=LABEL_ORDER, dtype=int)

    for _, row in df.iterrows():
        problem_leads = extract_problem_leads(row[dead_lead_column])
        if not problem_leads:
            continue

        if row["has_afib"]:
            for lead in problem_leads:
                nonworking.loc[lead, "AFIB"] += 1

        if row["has_norm"]:
            for lead in problem_leads:
                nonworking.loc[lead, "NORM"] += 1

    return nonworking


def build_nonworking_percent_df(
    nonworking_counts_df: pd.DataFrame,
    total_record_counts: pd.Series,
) -> pd.DataFrame:
    """
    Convert non-working lead counts into percentages for AFIB and NORM.
    """
    percent_df = nonworking_counts_df.astype(float).copy()

    for label in LABEL_ORDER:
        denom = float(total_record_counts[label])
        percent_df[label] = (percent_df[label] / denom) * 100.0 if denom > 0 else 0.0

    return percent_df

