# backend/main.py
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from io import BytesIO
import pdfplumber
import docx
import openai
import numpy as np

load_dotenv()  # loads .env in backend root if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Set it in .env or Render env vars.")
openai.api_key = OPENAI_API_KEY

app = FastAPI(title="DescriptionAI - Resume Processor")

# Update allowed_origins with your Vercel / domain names

allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://descriptionai-frontend.vercel.app",
    "https://descriptionai.online"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_resume_file(upload_file: UploadFile) -> str:
    # read bytes
    contents = upload_file.file.read()
    text = ""
    if upload_file.filename.lower().endswith(".pdf"):
        with pdfplumber.open(BytesIO(contents)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif upload_file.filename.lower().endswith(".docx") or upload_file.filename.lower().endswith(".doc"):
        doc = docx.Document(BytesIO(contents))
        for p in doc.paragraphs:
            text += p.text + "\n"
    else:
        # try as plain text
        try:
            text = contents.decode("utf-8", errors="ignore")
        except:
            text = ""
    return text.strip()

@app.get("/")
def root():
    return {"ok": True, "msg": "DescriptionAI backend alive"}

@app.post("/process_resume/")
async def process_resume(
    resume: UploadFile,
    job_desc: str = Form(...),
    email: str = Form(...)
):
    try:
        resume_text = parse_resume_file(resume)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not parse resume text")

        # embeddings (OpenAI)
        emb_res = openai.Embedding.create(model="text-embedding-3-small", input=resume_text)
        embedding_resume = emb_res["data"][0]["embedding"]

        emb_job = openai.Embedding.create(model="text-embedding-3-small", input=job_desc)
        embedding_job = emb_job["data"][0]["embedding"]

        # cosine similarity
        vec_r = np.array(embedding_resume, dtype=float)
        vec_j = np.array(embedding_job, dtype=float)
        cos_sim = float(np.dot(vec_r, vec_j) / (np.linalg.norm(vec_r) * np.linalg.norm(vec_j)))
        score = round(cos_sim * 100, 2)

        # suggestions via chat completion
        prompt = f"Job description:\n{job_desc}\n\nResume:\n{resume_text}\n\nProvide 5 short, actionable bullet suggestions to improve this resume for the job description."
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        suggestions = completion["choices"][0]["message"]["content"].strip()

        return {
            "match_score": score,
            "suggestions": suggestions,
            "email": email
        }

    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
