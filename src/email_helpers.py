from datetime import datetime, timezone
from typing import List, Dict, Any, Iterable, Optional, Tuple, Union
import json
import os
import pandas as pd
from googleapiclient.discovery import build, Resource
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from src.constants import REVIEW_CSV_COLUMNS, data_path

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def get_gmail_service(token_path: str = "token.json", creds_path: str = "credentials.json") -> Resource:
    """
    Authenticate with Gmail and return an authorized Gmail API service instance.

    This function loads an existing OAuth token if available, otherwise it
    initiates a browser-based login flow to request consent and generate a new token.
    The token is stored locally to avoid re-authentication in future runs.

    Args:
        token_path (str): Path to the stored OAuth token JSON file.
        creds_path (str): Path to the client credentials JSON file used
            to initiate the authentication flow when necessary.

    Returns:
        Resource: A Gmail API service client used to access user email data.
    """
    creds: Credentials | None = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def list_message_ids(
    service: Resource,
    max_results: int = 100,
    query: Optional[str] = None,
    page_token: Optional[str] = None,
    include_spam_trash: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Fetch a single page of Gmail message IDs, optionally filtered by a query.

    Args:
        service (Resource): An authenticated Gmail API service instance.
        max_results (int): Maximum number of email message entries to return.
        query (str, optional): Gmail search query string (e.g., "newer_than:30d").
        page_token (str, optional): Gmail page token for pagination.
        include_spam_trash (bool): Whether to include Spam/Trash in results.

    Returns:
        tuple[list[dict], str | None]: A list of message objects and the next page token.
    """
    results = (
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=max_results,
            q=query,
            pageToken=page_token,
            includeSpamTrash=include_spam_trash,
        )
        .execute()
    )
    return results.get("messages", []), results.get("nextPageToken")


def fetch_all_message_ids(
    service: Resource,
    total_limit: Optional[int] = None,
    page_size: int = 500,
    query: Optional[str] = None,
    include_spam_trash: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch all Gmail message IDs using pagination.

    Args:
        service (Resource): An authenticated Gmail API service instance.
        total_limit (int, optional): Stop after this many messages.
        page_size (int): Page size for Gmail API pagination.
        query (str, optional): Gmail search query string.
        include_spam_trash (bool): Whether to include Spam/Trash in results.

    Returns:
        list[dict]: A list of Gmail message objects containing `"id"` values.
    """
    all_messages: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    while True:
        remaining = None if total_limit is None else max(total_limit - len(all_messages), 0)
        if remaining == 0:
            break
        page_max = page_size if remaining is None else min(page_size, remaining)

        messages, page_token = list_message_ids(
            service,
            max_results=page_max,
            query=query,
            page_token=page_token,
            include_spam_trash=include_spam_trash,
        )
        all_messages.extend(messages)

        if not page_token or not messages:
            break

    return all_messages


def fetch_emails(
    service: Resource,
    max_results: int = 100,
    query: Optional[str] = None,
    page_token: Optional[str] = None,
    include_spam_trash: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch a batch of email metadata from the authenticated Gmail inbox.

    Retrieves message IDs for a subset of emails to support further
    fetching or processing. This does not download full email contents.

    Args:
        service (Resource): An authenticated Gmail API service instance.
        max_results (int): Maximum number of email message entries to return.
        query (str, optional): Gmail search query string (e.g., "newer_than:30d").
        page_token (str, optional): Gmail page token for pagination.
        include_spam_trash (bool): Whether to include Spam/Trash in results.

    Returns:
        list[dict]: A list of Gmail message objects containing `"id"` values
            for each email retrieved.
    """
    messages, _ = list_message_ids(
        service,
        max_results=max_results,
        query=query,
        page_token=page_token,
        include_spam_trash=include_spam_trash,
    )
    return messages


def _normalize_message_ids(emails: Iterable[Union[Dict[str, Any], str]]) -> List[str]:
    message_ids: List[str] = []
    for item in emails:
        if isinstance(item, str):
            message_ids.append(item)
        elif isinstance(item, dict) and "id" in item:
            message_ids.append(item["id"])
    return message_ids


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


def _format_sent_timestamp(
    date_header: Optional[str], internal_date_ms: Optional[Union[str, int]]
) -> str:
    """Format email sent timestamp: use Date header if present, else internalDate (epoch ms)."""
    if date_header and str(date_header).strip():
        return str(date_header).strip()
    if internal_date_ms is not None:
        try:
            ms = int(internal_date_ms)
            sec = ms / 1000.0
            return datetime.fromtimestamp(sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, OSError):
            return str(internal_date_ms)
    return ""


def export_emails(
    service: Resource,
    emails: Iterable[Union[Dict[str, Any], str]],
    filename: Optional[str] = None,
    append: bool = False,
    include_review_columns: bool = True,
    exclude_sender_emails: Optional[List[str]] = None,
) -> None:
    """
    Fetch email metadata and create a labeled dataset for manual review.

    For each message ID, full headers are retrieved and parsed to extract
    the Subject and From fields. A simple classifier scores each message,
    and the results are saved into a CSV for later inspection.

    Args:
        service (Resource): Authenticated Gmail API client.
        emails (list[dict] | list[str]): List of Gmail message objects or message IDs.
        filename (str): Name of the CSV file in which parsed results will be stored.
        append (bool): Append to existing CSV instead of overwriting.
        include_review_columns (bool): Add review columns (important/reason/etc) as blanks.
        exclude_sender_emails (list[str], optional): Do not export rows where From contains
            any of these addresses (e.g. your own email to exclude sent mail).

    Returns:
        int: Number of rows written to the CSV.
    """
    if filename is None:
        filename = data_path("emails_review.csv")
    email_data: List[Dict[str, Any]] = []
    message_ids = _normalize_message_ids(emails)
    exclude_lower = [e.strip().lower() for e in (exclude_sender_emails or []) if e]

    for msg_id in message_ids:
        msg = service.users().messages().get(userId="me", id=msg_id).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

        subject = headers.get("Subject", "")
        sender = headers.get("From", "")
        if exclude_lower and sender and any(ex in sender.lower() for ex in exclude_lower):
            continue
        date_sent = _format_sent_timestamp(headers.get("Date"), msg.get("internalDate"))

        row = {
            "message_id": msg_id,
            "subject": subject,
            "sender": sender,
            "date_sent": date_sent,
        }
        email_data.append(row)

    df = pd.DataFrame(email_data)
    if include_review_columns:
        df = _ensure_review_columns(df)

    n = len(df)
    if append:
        _append_dataframe_to_csv(df, filename)
    else:
        df.to_csv(filename, index=False)
    print(f"Exported {n} emails to {filename}")
    return n


def _get_user_email(service: Resource) -> Optional[str]:
    """Return the authenticated user's Gmail address, or None on error."""
    try:
        profile = service.users().getProfile(userId="me").execute()
        return (profile.get("emailAddress") or "").strip() or None
    except Exception:
        return None


def export_emails_paginated(
    service: Resource,
    filename: Optional[str] = None,
    query: Optional[str] = None,
    total_limit: Optional[int] = None,
    page_size: int = 500,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    inbox_only: bool = True,
    exclude_self: bool = True,
) -> None:
    """
    Export emails to CSV using Gmail pagination with checkpoint/resume support.

    By default only inbox messages are exported (no Sent/Drafts/etc.) and emails
    from yourself are excluded. Exports and checkpoints are stored under the data/ folder.
    """
    if filename is None:
        filename = data_path("emails_review.csv")
    if checkpoint_path is None:
        checkpoint_path = data_path("export_checkpoint.json")
    if inbox_only:
        base = (query or "").strip()
        query = f"in:inbox {base}".strip() if base else "in:inbox"
    exclude_sender_emails: Optional[List[str]] = None
    if exclude_self:
        user_email = _get_user_email(service)
        if user_email:
            exclude_sender_emails = [user_email]

    checkpoint = _load_checkpoint(checkpoint_path) if resume else {}
    page_token = checkpoint.get("next_page_token")
    exported = checkpoint.get("exported", 0)

    while True:
        remaining = None if total_limit is None else max(total_limit - exported, 0)
        if remaining == 0:
            break
        page_max = page_size if remaining is None else min(page_size, remaining)

        messages, next_token = list_message_ids(
            service,
            max_results=page_max,
            query=query,
            page_token=page_token,
        )
        if not messages:
            break

        n_written = export_emails(
            service,
            messages,
            filename=filename,
            append=os.path.exists(filename),
            exclude_sender_emails=exclude_sender_emails,
        )
        exported += n_written

        _save_checkpoint(
            checkpoint_path,
            {"next_page_token": next_token, "exported": exported, "query": query},
        )

        if not next_token:
            break
        page_token = next_token


def delete_emails_from_csv(
    service: Resource,
    df: Union[pd.DataFrame, str],
    decision_column: str = "decision",
    decision_value: str = "DELETE",
    dry_run: bool = False,
) -> None:
    """
    Move emails to Gmail Trash that have been labeled for deletion.

    This function looks for entries in the DataFrame where the decision equals
    `"DELETE"`, then attempts to delete each corresponding message ID.
    Progress and errors are logged to the console.

    Args:
        service (Resource): Gmail API client used to delete emails.
        df (pd.DataFrame | str): Labeled messages or path to the review CSV.
        decision_column (str): Column to check for delete decisions.
        decision_value (str): Value that triggers trashing (case-insensitive).
        dry_run (bool): If True, do not modify Gmail.

    Returns:
        None: Emails are moved to Trash in Gmail as side effects, with status logs printed.
    """
    if isinstance(df, str):
        df = pd.read_csv(df)

    if decision_column not in df.columns:
        print(f"Missing '{decision_column}' column. Nothing to delete.")
        return

    normalized = df[decision_column].astype(str).str.upper()
    to_delete = df[normalized == decision_value.upper()]
    print(f"Found {len(to_delete)} emails marked for deletion.")

    for _, row in to_delete.iterrows():
        msg_id = row.get("message_id")

        if pd.isna(msg_id):
            continue
        if dry_run:
            print(f"[DRY RUN] Would delete email: {row.get('subject')} from {row.get('sender')}")
            continue
        try:
            service.users().messages().trash(userId="me", id=msg_id).execute()
            print(f"Deleted email: {row.get('subject')} from {row.get('sender')}")
        except Exception as e:
            print(f"Error deleting {row.get('subject')}: {e}")
