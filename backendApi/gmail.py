import os
import base64
from typing import List
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


class GmailClient:
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self._authenticate()

    def _authenticate(self):
        """Authenticate and return the Gmail API service."""
        creds = None
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        return build('gmail', 'v1', credentials=creds)

    def _parse_email_content(self, parts):
        for part in parts:
            mime_type = part.get('mimeType', '')

            # Parsing für 'text/plain' und 'text/html'
            if mime_type in ['text/plain', 'text/html']:
                if 'body' in part and 'data' in part['body']:
                    data = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    if mime_type == 'text/plain':
                        return data
                    if mime_type == 'text/html':
                        soup = BeautifulSoup(data, 'html.parser')
                        return soup.get_text()  # HTML gefunden und geparst, Suche stoppen

        # Falls kein Inhalt gefunden, suche rekursiv
        for part in parts:
            if 'parts' in part:
                data = self._parse_email_content(part['parts'])
                if data:
                    return data

        return ''

    def fetch_emails(self, max_results, query=None, label_id=None) -> List[dict]:
        """Fetch emails from Gmail inbox."""
        try:
            results = self.service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q=query,
                labelIds=[label_id] if label_id else None
            ).execute()
            messages = results.get('messages', [])

            emails = []
            for message in messages:
                try:
                    msg = self.service.users().messages().get(userId='me', id=message['id']).execute()
                    payload = msg.get('payload', {})
                    headers = payload.get('headers', [])

                    subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
                    body = ''

                    if 'parts' in payload:
                        body = self._parse_email_content(payload['parts'])

                    emails.append({'title': subject, 'content': body})
                except Exception as e:
                    print(f"Fehler beim Verarbeiten der E-Mail mit der ID {message['id']}: {str(e)}")

            return emails
        except HttpError as error:
            print(f'Ein Fehler ist aufgetreten: {error}')
            return []
