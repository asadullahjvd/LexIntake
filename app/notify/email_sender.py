"""
Email notifications — sends the client an email when their case has a
work-related deadline attached.

Uses plain SMTP (works with Gmail/Outlook/any provider via an app
password) so there's no new third-party account to set up. If you later
want a transactional email API instead (SendGrid, Resend, SES), swap the
body of send_email() only — every caller in the app depends solely on
send_email(to_email, subject, body), so nothing else needs to change.

Required environment variables (put these in a .env file and load it
with python-dotenv, or set them in your deployment environment):
    SMTP_HOST      e.g. "smtp.gmail.com"
    SMTP_PORT      e.g. 587
    SMTP_USER      the sending mailbox, e.g. "yourfirm@gmail.com"
    SMTP_PASSWORD  an APP PASSWORD, not your normal account password
                   (Gmail: Google Account -> Security -> App Passwords)
    FROM_EMAIL     optional, defaults to SMTP_USER
"""
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(to_email: str, subject: str, body: str) -> bool:
    """Sends a plain-text email. Returns True on success, False on any
    failure — this never raises, so a broken mail config or a down SMTP
    server can't crash the graph run. Check the return value / logs if
    you need to know whether it actually went out."""
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    from_email = os.environ.get("FROM_EMAIL", user)

    if not all([host, user, password, to_email]):
        print(f"[email_sender] Missing SMTP config or recipient — skipping send to {to_email!r}")
        return False

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_email, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[email_sender] Failed to send to {to_email!r}: {e}")
        return False