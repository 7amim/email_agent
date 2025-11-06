from typing import List, Dict, Any
import os
import pandas as pd
from googleapiclient.discovery import build, Resource
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

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


def fetch_emails(service: Resource, max_results: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch a batch of email metadata from the authenticated Gmail inbox.

    Retrieves message IDs for a subset of emails to support further
    fetching or processing. This does not download full email contents.

    Args:
        service (Resource): An authenticated Gmail API service instance.
        max_results (int): Maximum number of email message entries to return.

    Returns:
        list[dict]: A list of Gmail message objects containing `"id"` values
            for each email retrieved.
    """
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    return results.get("messages", [])


def export_emails(service: Resource, emails: List[Dict[str, Any]], filename: str = "emails_review.csv") -> None:
    """
    Fetch email metadata and create a labeled dataset for manual review.

    For each message ID, full headers are retrieved and parsed to extract
    the Subject and From fields. A simple classifier scores each message,
    and the results are saved into a CSV for later inspection.

    Args:
        service (Resource): Authenticated Gmail API client.
        emails (list[dict]): List of Gmail message objects where each item contains `"id"`.
        filename (str): Name of the CSV file in which parsed results will be stored.

    Returns:
        None: Results are written directly to a CSV, with the number of exported
            emails printed to the console.
    """
    email_data = []

    for e in emails:
        msg = service.users().messages().get(userId="me", id=e["id"]).execute()
        headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}

        subject = headers.get("Subject", "")
        sender = headers.get("From", "")

        email_data.append({
            "Message_ID": e["id"],
            "Subject": subject,
            "Sender": sender
        })

    df = pd.DataFrame(email_data)
    df.to_csv(filename, index=False)
    print(f"Exported {len(df)} emails to {filename}")


def delete_emails_from_csv(service: Resource, df: pd.DataFrame) -> None:
    """
    Remove emails permanently from Gmail that have been labeled for deletion.

    This function looks for entries in the DataFrame where the decision equals
    `"DELETE"`, then attempts to delete each corresponding message ID.
    Progress and errors are logged to the console.

    Args:
        service (Resource): Gmail API client used to delete emails.
        df (pd.DataFrame): A table of labeled messages, containing columns:
            `"Message_ID"`, `"Subject"`, `"Sender"`, and `"Decision"`.

    Returns:
        None: Emails are deleted in Gmail as side effects, with status logs printed.
    """
    to_delete = df[df["Decision"].str.upper() == "DELETE"]
    print(f"Found {len(to_delete)} emails marked for deletion.")

    for _, row in to_delete.iterrows():
        msg_id = row.get("Message_ID")

        if pd.isna(msg_id):
            continue
        try:
            service.users().messages().trash(userId="me", id=msg_id).execute()
            print(f"Deleted email: {row['Subject']} from {row['Sender']}")
        except Exception as e:
            print(f"Error deleting {row['Subject']}: {e}")
