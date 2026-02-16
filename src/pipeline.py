import json
import os
from typing import Optional, Dict, Any, List

import pandas as pd

from src.agent.email_agent import EmailAgent
from src.constants import REVIEW_CSV_COLUMNS, CONFIDENCE_RANK

PROMO_SUBJECT_KEYWORDS = [
    "newsletter",
    "unsubscribe",
    "promo",
    "promotion",
    "sale",
    "discount",
    "offer",
    "deal",
    "marketing",
    "digest",
]
PROMO_SENDER_KEYWORDS = ["no-reply", "noreply", "newsletter", "marketing", "promo"]


def _ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in REVIEW_CSV_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def _append_dataframe_to_csv(df: pd.DataFrame, filename: str) -> None:
    file_exists = os.path.exists(filename)
    df.to_csv(filename, index=False, mode="a" if file_exists else "w", header=not file_exists)


def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    with open(checkpoint_path, "r") as handle:
        return json.load(handle)


def _save_checkpoint(checkpoint_path: str, payload: Dict[str, Any]) -> None:
    if not checkpoint_path:
        return
    with open(checkpoint_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _apply_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    subject = df["Subject"].astype(str).str.lower()
    sender = df["Sender"].astype(str).str.lower()

    subject_match = subject.apply(lambda text: any(k in text for k in PROMO_SUBJECT_KEYWORDS))
    sender_match = sender.apply(lambda text: any(k in text for k in PROMO_SENDER_KEYWORDS))
    heuristic_mask = subject_match | sender_match

    df.loc[heuristic_mask, "important"] = "No"
    df.loc[heuristic_mask, "reason"] = "Heuristic: promotional/newsletter pattern"
    df.loc[heuristic_mask, "confidence"] = "Medium"
    return df


def _flag_low_confidence(df: pd.DataFrame, threshold: str) -> pd.DataFrame:
    threshold_rank = CONFIDENCE_RANK.get(threshold.lower())
    if threshold_rank is None:
        return df
    confidence_rank = df["confidence"].astype(str).str.lower().map(CONFIDENCE_RANK)
    df["Needs_Review"] = confidence_rank.fillna(0) <= threshold_rank
    return df


async def classify_csv_in_batches(
    agent: EmailAgent,
    input_csv: str,
    output_csv: str = "classified_emails.csv",
    batch_size: int = 200,
    checkpoint_path: str = "classify_checkpoint.json",
    resume: bool = True,
    apply_heuristics: bool = True,
    confidence_flag_threshold: Optional[str] = "Medium",
) -> None:
    """
    Classify emails from a CSV in batches and append results to a new CSV.
    """
    checkpoint = _load_checkpoint(checkpoint_path) if resume else {}
    start_chunk = checkpoint.get("next_chunk_index", 0)
    processed_rows = checkpoint.get("processed_rows", 0)

    reader = pd.read_csv(input_csv, chunksize=batch_size)

    for chunk_index, chunk in enumerate(reader):
        if chunk_index < start_chunk:
            continue

        chunk = _ensure_review_columns(chunk)

        if apply_heuristics:
            chunk = _apply_heuristics(chunk)

        needs_llm = chunk["important"].astype(str).str.strip() == ""
        if needs_llm.any():
            llm_input = chunk[needs_llm][["Subject", "Sender"]].copy()
            llm_result = await agent.run(llm_input)
            chunk.loc[needs_llm, ["important", "reason", "confidence"]] = llm_result[
                ["important", "reason", "confidence"]
            ].values

        if confidence_flag_threshold:
            chunk = _flag_low_confidence(chunk, confidence_flag_threshold)

        _append_dataframe_to_csv(chunk, output_csv)
        processed_rows += len(chunk)

        _save_checkpoint(
            checkpoint_path,
            {
                "next_chunk_index": chunk_index + 1,
                "processed_rows": processed_rows,
                "input_csv": input_csv,
                "output_csv": output_csv,
            },
        )

