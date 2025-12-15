import io
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz
from google.generativeai import configure, GenerativeModel
import json
import re
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="Diagnostics AI Parser (Strict Extraction)")

API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=API_KEY)

model = GenerativeModel("gemini-2.5-flash")

class ParseRequest(BaseModel):
    pdfUrl: str


RANGES = {
    "hemoglobin": (12, 16),
    "fastingGlucose": (70, 99),
    "hdl": (40, 200),
    "ldl": (0, 100),
    "triglycerides": (0, 150),
    "tsh": (0.4, 4.0),
    "vitaminD": (30, 100),
    "alt": (7, 56),
    "ast": (10, 40)
}


def calculate_status(name, value):
    if value is None:
        return None
    low, high = RANGES[name]
    return "bad" if value < low or value > high else "good"


def download_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.content


def pdf_to_images(pdf_bytes: bytes, max_pages: int | None = None):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_num in range(pages_to_process):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        images.append(buf.getvalue())

    doc.close()
    return images


def classify_report_is_blood(img_buffers: list[bytes]) -> dict:
    prompt = """
You are a medical document classifier.

Based ONLY on the document content,
determine whether this is a BLOOD TEST REPORT.

Return JSON only:
{
  "isBloodReport": true | false,
  "confidence": "low" | "medium" | "high",
  "reason": "short explanation"
}
"""

    response = model.generate_content(
        [prompt, *[{"mime_type": "image/jpeg", "data": img} for img in img_buffers]]
    )

    try:
        text_json = re.search(r"\{.*\}", response.text, re.DOTALL).group(0)
        return json.loads(text_json)
    except:
        raise HTTPException(500, "Failed to classify report type")

@app.get("/")
def root():
    return {"message": "Diagnostics AI Parser is running."}


@app.post("/parse-report")
def parse_report(req: ParseRequest):

    print("/parse-report request received")
    
    try:
        pdf_bytes = download_pdf(req.pdfUrl)
    except:
        raise HTTPException(400, "Could not download PDF")

    
    preview_images = pdf_to_images(pdf_bytes, max_pages=2)

    
    classification = classify_report_is_blood(preview_images)

    if not classification.get("isBloodReport"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type. Reason: {classification.get('reason')}"
        )

    
    full_images = pdf_to_images(pdf_bytes)

    
    prompt = """
Extract ONLY these biomarkers from the blood report:

hemoglobin, fasting glucose, hdl, ldl, triglycerides, tsh, vitamin d, alt, ast.

STRICT RULES:
- Extract the EXACT numeric value ONLY if it is explicitly written.
- If mentioned WITHOUT a numeric value, return null.
- If NOT mentioned at all, return null.
- Do NOT infer, estimate, guess, or approximate.
- Do NOT use medical knowledge.

Return JSON only in this format:
{
  "hemoglobin": number | null,
  "fastingGlucose": number | null,
  "hdl": number | null,
  "ldl": number | null,
  "triglycerides": number | null,
  "tsh": number | null,
  "vitaminD": number | null,
  "alt": number | null,
  "ast": number | null
}
"""

    response = model.generate_content(
        [prompt, *[{"mime_type": "image/jpeg", "data": img} for img in full_images]]
    )

    
    try:
        text_json = re.search(r"\{.*\}", response.text, re.DOTALL).group(0)
        values = json.loads(text_json)
    except:
        raise HTTPException(500, "AI did not return valid JSON")
    
    
    missing_biomarkers = [k for k, v in values.items() if v is None]

    
    final = {}
    for key, value in values.items():
        if value is None:
            final[key] = {"value": None, "status": None}
        else:
            final[key] = {
                "value": float(value),
                "status": calculate_status(key, float(value))
            }

    
    return {
        "reportTypeConfidence": classification.get("confidence"),
        "missingBiomarkers": missing_biomarkers,
        "biomarkers": final
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "parser_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )

