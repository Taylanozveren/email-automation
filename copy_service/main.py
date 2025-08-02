"""
copy_service/main.py
POST /generate-email -> Hugging Face (featherless-ai) ile kişiye özel e-posta metni üretir
Swagger: /docs  (root "/" otomatik /docs'a yönlendirir)
Health:  /healthz
Meta:    /meta
"""

import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# .env yükle
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV, override=True)

HF_MODEL: str    = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
HF_PROVIDER: str = os.getenv("HF_PROVIDER", "featherless-ai")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

client = InferenceClient(
    model=HF_MODEL,
    provider=HF_PROVIDER,
    token=HF_TOKEN
)

app = FastAPI(title="KO Copy Service", version="0.4.0")

class LeadData(BaseModel):
    name: str
    email: str
    segment: str

# ---- Helpers ---------------------------------------------------------------

def build_prompt(d: LeadData) -> str:
    """LLM için sıkı kurallı prompt."""
    return (
        "You are a friendly English learning coach at Konuşarak Öğren.\n"
        f"Write a concise (<=120 words) follow-up email to {d.name} "
        f"(segment: {d.segment}). Tone: warm, motivating, conversational.\n"
        "Rules:\n"
        "- Exactly ONE clear CTA link, no extra links.\n"
        "- No dialogue markers, no prompt repetition, no YAML/JSON/markdown.\n"
        "- First line must read like a natural subject line (<=55 chars).\n"
        "- Body in plain text, short paragraphs.\n"
        f"CTA link (must appear once): https://konusarakogren.com/activate?e={d.email}\n\n"
        "Return only the email text starting with the subject line."
    )

def extract_subject(body: str) -> str:
    """İlk satırdan konuyu çıkarır ve 55 karakterle sınırlar."""
    if not body:
        return "Follow up from Konuşarak Öğren"
    first = body.splitlines()[0].strip()
    if first.lower().startswith("subject:"):
        first = first.split(":", 1)[1].strip()
    first = re.sub(r"\s+", " ", first)
    if len(first) < 3 or len(first.split()) < 2:
        return "Follow up from Konuşarak Öğren"
    return first[:55]

def clean_body(raw: str, email: str) -> str:
    """
    - 'Subject:' satırını gövdeden atar
    - Boşlukları normalize eder
    - Tüm linkleri temizler ve TEK standart CTA ekler
    - Uzunluğu savunmacı biçimde sınırlar
    """
    if not raw:
        raw = ""

    lines = [l.rstrip() for l in raw.splitlines()]
    if lines and lines[0].lower().startswith("subject:"):
        lines = lines[1:]

    body = "\n".join(lines).strip()

    # Fazla boş satırları azalt
    while "\n\n\n" in body:
        body = body.replace("\n\n\n", "\n\n")

    # Tüm linkleri (http/https) temizle (tek CTA kuralı için)
    body = re.sub(r"https?://\S+", "", body).strip()
    if body and not body.endswith("\n"):
        body += "\n"

    # Standart CTA'yı tek sefer ekle
    cta = f"https://konusarakogren.com/activate?e={email}"
    body += f"\n👉 Hemen başla: {cta}"

    # Sert uzunluk sınırı (savunmacı)
    if len(body) > 1200:
        body = body[:1180].rsplit(" ", 1)[0] + " ..."

    # İki ve daha fazla boş satırı teke indir
    body = re.sub(r"\n{3,}", "\n\n", body)

    return body.strip()

# ---- Routes ----------------------------------------------------------------

# Root'u doğrudan Swagger'a yönlendir
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs", status_code=307)

# Uptime/health check
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}

# Eski root bilgisinin eşleniği (diagnostic)
@app.get("/meta", include_in_schema=False)
def meta():
    return {
        "status": "ok",
        "model": HF_MODEL,
        "provider": HF_PROVIDER
    }

@app.post("/generate-email")
async def generate_email(data: LeadData):
    prompt = build_prompt(data)
    try:
        resp = client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        # LLM çağrısı hatalarında 502 dön
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    raw = (resp.choices[0].message.content or "").strip()
    subject = extract_subject(raw)
    body = clean_body(raw, data.email)

    return {
        "subject": subject,
        "body": body,
        "model": HF_MODEL,
        "provider": HF_PROVIDER,
        "email": data.email,
        "name": data.name,
        "segment": data.segment
    }
