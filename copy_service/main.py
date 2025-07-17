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

# .env dosyasını yükle
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV, override=True)

HF_MODEL    = os.getenv("HF_MODEL")
HF_PROVIDER = os.getenv("HF_PROVIDER")          # featherless-ai
HF_TOKEN    = os.getenv("HF_TOKEN")             # Boşsa header eklenmez

client = InferenceClient(
    model=HF_MODEL,
    provider=HF_PROVIDER,
    token=HF_TOKEN
)

app = FastAPI(title="KO Copy Service", version="0.2.0")

class LeadData(BaseModel):
    name: str
    email: str
    segment: str

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": HF_MODEL,
        "provider": HF_PROVIDER
    }

@app.post("/generate-email")
async def generate_email(data: LeadData):
    prompt = (
        "You are a friendly English learning coach at Konuşarak Öğren.\n"
        f"Write a concise (<120 words) follow-up email to {data.name} "
        f"(segment: {data.segment}). Use a warm, motivating tone and include "
        "exactly ONE CTA link:\n"
        f"https://konusarakogren.com/activate?e={data.email}\n\nEmail:\n"
    )

    # Featherless-AI endpoint: chat.completions
    resp = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    email_body = resp.choices[0].message.content.strip()
    subject = email_body.splitlines()[0][:60] or "Follow up from Konuşarak Öğren"

    return {
        "subject": subject,
        "body": email_body,
        "model": HF_MODEL,
        "provider": HF_PROVIDER
    }
