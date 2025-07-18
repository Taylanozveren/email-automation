"""
copy_service/main.py
POST /generate-email  -> Hugging Face (featherless-ai) ile kişiye özel e-posta metni üretir
"""

import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# .env yükle
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV, override=True)

HF_MODEL    = os.getenv("HF_MODEL")
HF_PROVIDER = os.getenv("HF_PROVIDER")
HF_TOKEN    = os.getenv("HF_TOKEN")

client = InferenceClient(
    model=HF_MODEL,
    provider=HF_PROVIDER,
    token=HF_TOKEN
)

app = FastAPI(title="KO Copy Service", version="0.3.0")

class LeadData(BaseModel):
    name: str
    email: str
    segment: str

def build_prompt(d: LeadData) -> str:
    return (
        "You are a friendly English learning coach at Konuşarak Öğren.\n"
        f"Write a concise (<=120 words) follow-up email to {d.name} "
        f"(segment: {d.segment}). Tone: warm, motivating, conversational.\n"
        "Rules:\n"
        "- Exactly ONE clear CTA link, no extra links.\n"
        "- No dialogue markers, no prompts repetition.\n"
        "- First line should read like a natural subject line (<=55 chars).\n"
        "- Body in plain text (no markdown), short paragraphs.\n"
        f"CTA link (must appear once): https://konusarakogren.com/activate?e={d.email}\n\n"
        "Return only the email content starting with the subject line.\n"
    )

def extract_subject(body: str) -> str:
    if not body:
        return "Follow up from Konuşarak Öğren"
    first_line = body.splitlines()[0].strip()
    # Remove leading "Subject:" if present
    if first_line.lower().startswith("subject:"):
        first_line = first_line.split(":", 1)[1].strip()
    # Basic cleanup
    first_line = first_line.replace("  ", " ").strip()
    # Fallback if too short / weird
    if len(first_line) < 3 or len(first_line.split()) < 2:
        return "Follow up from Konuşarak Öğren"
    return first_line[:55]

def clean_body(raw: str) -> str:
    if not raw:
        return ""
    lines = [l.rstrip() for l in raw.splitlines()]
    # Drop potential first line if it’s identical to subject pattern like “Subject: …”
    if lines and lines[0].lower().startswith("subject:"):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    # Collapse excessive blank lines
    while "\n\n\n" in body:
        body = body.replace("\n\n\n", "\n\n")
    # Ensure only one CTA link – remove duplicates if model hallucinated
    import re
    cta_pattern = re.compile(r"https://konusarakogren\.com/activate\?e=[^\s]+", re.IGNORECASE)
    all_ctas = cta_pattern.findall(body)
    if len(all_ctas) > 1:
        # Keep first, remove others
        first = all_ctas[0]
        body = cta_pattern.sub("", body)
        body += f"\n\n👉 Başla: {first}"
    elif len(all_ctas) == 1:
        # Standardize format
        first = all_ctas[0]
        if "👉" not in body:
            body += f"\n\n👉 Başla: {first}"
    else:
        # No CTA found → append
        body += f"\n\n👉 Başla: https://konusarakogren.com/activate?e=placeholder"
    # Trim hard length cap (defensive)
    if len(body) > 1200:
        body = body[:1180].rsplit(" ", 1)[0] + " ..."
    return body.strip()

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": HF_MODEL,
        "provider": HF_PROVIDER
    }

@app.post("/generate-email")
async def generate_email(data: LeadData):
    prompt = build_prompt(data)

    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = resp.choices[0].message.content.strip()
    subject = extract_subject(raw)
    body = clean_body(raw)

    # ECHO alanları ekliyoruz (kritik)
    return {
        "subject": subject,
        "body": body,
        "model": HF_MODEL,
        "provider": HF_PROVIDER,
        "email": data.email,
        "name": data.name,
        "segment": data.segment
    }
