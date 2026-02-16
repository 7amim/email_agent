"""
Email classification review UI (NiceGUI).

Loads data/classified_emails.csv (or data/emails_review.csv as fallback),
displays a filterable grid, and saves edits back to data/classified_emails.csv.
Run: python app.py
"""

import os
from typing import Optional

import pandas as pd
from nicegui import ui

from src.constants import REVIEW_CSV_COLUMNS, data_path


def _ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in REVIEW_CSV_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def _load_data():
    """Load CSV from data/classified_emails.csv or fallback to data/emails_review.csv."""
    classified_path = data_path("classified_emails.csv")
    review_path = data_path("emails_review.csv")
    if os.path.exists(classified_path):
        df = pd.read_csv(classified_path)
        target = classified_path
    elif os.path.exists(review_path):
        df = pd.read_csv(review_path)
        df = _ensure_review_columns(df)
        target = classified_path
    else:
        df = pd.DataFrame(columns=REVIEW_CSV_COLUMNS)
        target = classified_path
    return df, target


def _df_to_row_data(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts for AG Grid, normalizing NaN for JSON."""
    out = df.fillna("").astype(str).where(df.notna(), "").to_dict("records")
    for row in out:
        for k, v in row.items():
            if pd.isna(v) or (isinstance(v, float) and pd.isna(v)):
                row[k] = ""
    return out


def _make_column_defs():
    """AG Grid column definitions: read-only metadata, editable classification fields."""
    return [
        {"field": "message_id", "headerName": "Message ID", "editable": False, "width": 140},
        {"field": "subject", "headerName": "Subject", "editable": False, "flex": 1},
        {"field": "sender", "headerName": "Sender", "editable": False, "width": 200},
        {"field": "date_sent", "headerName": "Date sent", "editable": False, "width": 180},
        {
            "field": "important",
            "headerName": "Important",
            "editable": True,
            "width": 100,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {"values": ["Yes", "No", ""]},
        },
        {"field": "reason", "headerName": "Reason", "editable": True, "flex": 1},
        {
            "field": "confidence",
            "headerName": "Confidence",
            "editable": True,
            "width": 120,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {"values": ["High", "Medium", "Low", ""]},
        },
        {"field": "needs_review", "headerName": "Needs review", "editable": False, "width": 110},
        {"field": "suggested_decision", "headerName": "Suggested", "editable": False, "width": 100},
        {
            "field": "decision",
            "headerName": "Decision",
            "editable": True,
            "width": 100,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {"values": ["DELETE", "KEEP", "REVIEW", ""]},
        },
        {"field": "notes", "headerName": "Notes", "editable": True, "width": 150},
    ]


def _apply_filters(
    df: pd.DataFrame,
    confidence_val: Optional[str],
    decision_val: Optional[str],
    important_val: Optional[str],
    search_val: str,
) -> pd.DataFrame:
    out = df.copy()
    if confidence_val:
        out = out[out["confidence"].astype(str).str.strip().str.lower() == confidence_val.lower()]
    if decision_val:
        if decision_val == "(blank)":
            out = out[out["decision"].astype(str).str.strip() == ""]
        else:
            out = out[out["decision"].astype(str).str.strip().str.upper() == decision_val.upper()]
    if important_val:
        out = out[out["important"].astype(str).str.strip().str.lower() == important_val.lower()]
    search = (search_val or "").strip()
    if search:
        s = search.lower()
        mask = (
            out["subject"].astype(str).str.lower().str.contains(s, na=False)
            | out["sender"].astype(str).str.lower().str.contains(s, na=False)
        )
        out = out[mask]
    return out


@ui.page("/")
def index():
    df, target_path = _load_data()
    # Normalize for display (avoid NaN in JSON)
    for c in df.columns:
        df[c] = df[c].astype(object).where(df[c].notna(), "")

    with ui.column().classes("w-full gap-4 p-4"):
        ui.label("Email classification review").classes("text-2xl font-bold")

        with ui.row().classes("items-center gap-4 flex-wrap"):
            ui.button("Refresh", on_click=lambda: _do_refresh()).props("outline")
            confidence_select = ui.select(
                ["", "High", "Medium", "Low"],
                value="",
                label="Confidence",
            ).classes("w-40")
            decision_select = ui.select(
                ["", "DELETE", "KEEP", "REVIEW", "(blank)"],
                value="",
                label="Decision",
            ).classes("w-40")
            important_select = ui.select(
                ["", "Yes", "No"],
                value="",
                label="Important",
            ).classes("w-32")
            search_input = ui.input(placeholder="Search subject/sender").classes("w-56")

        container = ui.column().classes("w-full")

    def update_grid():
        c = confidence_select.value
        d = decision_select.value
        i = important_select.value
        s = search_input.value or ""
        filtered = _apply_filters(df, c or None, d or None, i or None, s)
        row_data = _df_to_row_data(filtered)
        if hasattr(container, "aggrid_ref") and container.aggrid_ref:
            container.aggrid_ref.options["rowData"] = row_data
            container.aggrid_ref.update()

    def on_cell_change(e):
        new_row = e.args.get("data") or {}
        mid = new_row.get("message_id")
        if mid is None or mid == "":
            return
        idx = df[df["message_id"].astype(str) == str(mid)].index
        if len(idx) == 0:
            return
        for col in ["important", "reason", "confidence", "decision", "notes"]:
            if col in new_row:
                df.loc[idx[0], col] = new_row[col] if new_row[col] != "" else ""
        df.to_csv(target_path, index=False)
        ui.notify("Saved", type="positive")
        # Keep grid rowData in sync
        if hasattr(container, "aggrid_ref") and container.aggrid_ref:
            rd = container.aggrid_ref.options.get("rowData") or []
            container.aggrid_ref.options["rowData"] = [
                (row | new_row) if str(row.get("message_id")) == str(mid) else row for row in rd
            ]
            container.aggrid_ref.update()

    def _do_refresh():
        nonlocal df
        df, _ = _load_data()
        for c in df.columns:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
        confidence_select.value = ""
        decision_select.value = ""
        important_select.value = ""
        search_input.value = ""
        filtered = _apply_filters(df, None, None, None, "")
        row_data = _df_to_row_data(filtered)
        if hasattr(container, "aggrid_ref") and container.aggrid_ref:
            container.aggrid_ref.options["rowData"] = row_data
            container.aggrid_ref.update()
        ui.notify("Reloaded from disk")

    for el in [confidence_select, decision_select, important_select]:
        el.on("update:model-value", lambda: update_grid())
    search_input.on("input", lambda: update_grid())

    filtered = _apply_filters(df, None, None, None, "")
    row_data = _df_to_row_data(filtered)
    aggrid = ui.aggrid(
        {
            "columnDefs": _make_column_defs(),
            "rowData": row_data,
            "defaultColDef": {"sortable": True, "resizable": True},
            "pagination": True,
            "paginationPageSize": 50,
            "paginationPageSizeSelector": [25, 50, 100, 200],
            "stopEditingWhenCellsLoseFocus": True,
            "domLayout": "normal",
        },
    ).classes("w-full").on("cellValueChanged", on_cell_change)
    container.aggrid_ref = aggrid


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Email classification review", reload=False)
